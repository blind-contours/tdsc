"""Nuisance-misspecification scenario grid.

Scenarios (DGP confounders: propensity uses x1,x2,x3; event hazard uses
x1..x4 with effect modifier tau(x1,x3); censoring uses x1,x2,x5,A):

  well        -- every nuisance sees all covariates.
  outcome_mis -- outcome hazards blinded to x1 and x3 (main confounder +
                 effect modifier); weight models (g, S_c) correct.
                 DR prediction: EIF-corrected estimators stay ~unbiased,
                 naive plug-in does not; also the stress test for targeting's
                 gradient span and the one-step residual top-up.
  weights_mis -- propensity and censoring nets blinded to x1 and x2; outcome
                 hazards correct. DR prediction: corrected estimators stay
                 ~unbiased through the hazard arm.

The propensity is always fit by a standalone net (never the trunk head) so the
outcome and weight models can be blinded independently.

Usage: python scripts/run_scenarios.py --scenario outcome_mis --reps 150
Results append to results/mc_scen_<scenario>.npz (resumable via --start).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tdsc import dgp
from tdsc.bands import if_se, simultaneous_band
from tdsc.estimators import isotonize, km_by_arm, naive_plugin, one_step, tda
from tdsc.influence import contrast_estimates, contrast_ifs
from tdsc.metrics import summarize_curve, summarize_scalar
from tdsc.model import nuisances, train_censnet, train_dragonsurv, train_propnet

# covariate indices: x1=0, x2=1, x3=2, x4=3, x5=4
ALL = None
SCENARIOS = {
    "well": dict(cols_out=ALL, cols_w=ALL),
    "outcome_mis": dict(cols_out=[1, 3, 4, 5, 6, 7, 8, 9], cols_w=ALL),
    "weights_mis": dict(cols_out=ALL, cols_w=[2, 3, 4, 5, 6, 7, 8, 9]),
}


def run_rep(rep, n, K, epochs, cols_out, cols_w):
    data = dgp.simulate(n, K=K, seed=1000 + rep)
    net = train_dragonsurv(data, max_epochs=epochs, seed=rep, cols=cols_out)
    censnet = train_censnet(data, max_epochs=epochs, seed=rep + 5000, cols=cols_w)
    propnet = train_propnet(data, max_epochs=epochs, seed=rep + 9000, cols=cols_w)
    h1, h0, g, Sc_lag = nuisances(net, censnet, data, propnet=propnet)

    psi_naive = naive_plugin(h1, h0)
    psi_os, D_os = one_step(data, h1, h0, g, Sc_lag)
    psi_km = km_by_arm(data)
    psi_ipcw = km_by_arm(data, adjusted=True, g=g, Sc_lag=Sc_lag)
    res = tda(net, censnet, data, propnet=propnet)
    psi_tda, D = res["psi"], res["D"]
    unsolved = np.abs(res["final_pnd"]) > res["final_tol"]
    psi_topup = psi_tda + unsolved * res["final_pnd"]

    D_ben, D_rmst = contrast_ifs(D)
    D_ben_os, D_rmst_os = contrast_ifs(D_os)
    crit, _ = simultaneous_band(D_ben, seed=rep)
    return dict(
        ben_naive=contrast_estimates(psi_naive)[0],
        ben_os=contrast_estimates(psi_os)[0],
        ben_os_iso=contrast_estimates(isotonize(psi_os))[0],
        ben_ipcw=contrast_estimates(psi_ipcw)[0],
        ben_km=contrast_estimates(psi_km)[0],
        ben_tda=contrast_estimates(psi_tda)[0],
        ben_tda_topup=contrast_estimates(psi_topup)[0],
        se_ben_tda=if_se(D_ben),
        se_ben_os=if_se(D_ben_os),
        crit_tda=crit,
        rmst_naive=contrast_estimates(psi_naive)[1],
        rmst_os=contrast_estimates(psi_os)[1],
        rmst_tda=contrast_estimates(psi_tda)[1],
        rmst_tda_topup=contrast_estimates(psi_topup)[1],
        se_rmst_tda=D_rmst.std(ddof=1) / np.sqrt(n),
        se_rmst_os=D_rmst_os.std(ddof=1) / np.sqrt(n),
        n_tda_iters=len(res["history"]),
        tda_converged=float(res["converged"]),
        n_unsolved_coords=float(unsolved.sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    cfg = SCENARIOS[args.scenario]

    resdir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(resdir, exist_ok=True)
    path = os.path.join(resdir, f"mc_scen_{args.scenario}.npz")
    store = {k: list(v) for k, v in np.load(path).items()} if os.path.exists(path) else {}

    truth = dgp.true_estimands(K=args.K)
    for rep in range(args.start, args.start + args.reps):
        out = run_rep(rep, args.n, args.K, args.epochs, cfg["cols_out"], cfg["cols_w"])
        for k, v in out.items():
            store.setdefault(k, []).append(v)
        np.savez(path, **{k: np.asarray(v) for k, v in store.items()})
        if (rep - args.start + 1) % 10 == 0 or rep == args.start:
            print(f"[{args.scenario}] rep {rep}: done ({len(store['rmst_tda'])} stored)",
                  flush=True)

    S = {k: np.asarray(v) for k, v in store.items()}
    ben_true = truth["benefit"]
    R = S["ben_tda"].shape[0]
    d = ((S["ben_tda"] - ben_true) ** 2).mean(axis=1) - \
        ((S["ben_os"] - ben_true) ** 2).mean(axis=1)
    summary = {
        "scenario": args.scenario,
        "benefit_curve": {
            "naive": summarize_curve(S["ben_naive"], ben_true),
            "km_confounded": summarize_curve(S["ben_km"], ben_true),
            "ipcw_km": summarize_curve(S["ben_ipcw"], ben_true),
            "one_step": summarize_curve(S["ben_os"], ben_true, se=S["se_ben_os"]),
            "one_step_isotonized": summarize_curve(S["ben_os_iso"], ben_true),
            "tda": summarize_curve(S["ben_tda"], ben_true, se=S["se_ben_tda"]),
            "tda_topup": summarize_curve(S["ben_tda_topup"], ben_true, se=S["se_ben_tda"]),
            "tda_simultaneous": summarize_curve(S["ben_tda"], ben_true,
                                                se=S["se_ben_tda"], crit=S["crit_tda"]),
        },
        "rmst_diff": {
            "naive": summarize_scalar(S["rmst_naive"], truth["rmst_diff"]),
            "one_step": summarize_scalar(S["rmst_os"], truth["rmst_diff"], se=S["se_rmst_os"]),
            "tda": summarize_scalar(S["rmst_tda"], truth["rmst_diff"], se=S["se_rmst_tda"]),
            "tda_topup": summarize_scalar(S["rmst_tda_topup"], truth["rmst_diff"],
                                          se=S["se_rmst_tda"]),
        },
        "paired_mse_diff_tda_minus_onestep": dict(
            mean=float(d.mean()), mc_se=float(d.std(ddof=1) / np.sqrt(R))),
        "targeting": dict(mean_iters=float(S["n_tda_iters"].mean()),
                          frac_converged=float(S["tda_converged"].mean()),
                          mean_unsolved_coords=float(S["n_unsolved_coords"].mean())),
        "reps": int(R),
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(resdir, f"summary_scen_{args.scenario}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
