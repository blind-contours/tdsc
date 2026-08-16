"""Monte Carlo study: bias / variance / MSE / coverage across estimators.

Usage: python scripts/run_montecarlo.py [--reps 200] [--n 1000] [--K 30]
Results are appended to results/mc_<tag>.npz so long runs can be resumed/split.
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
from tdsc.model import nuisances, train_censnet, train_dragonsurv


def run_rep(rep, n, K, epochs):
    data = dgp.simulate(n, K=K, seed=1000 + rep)
    net = train_dragonsurv(data, max_epochs=epochs, seed=rep)
    censnet = train_censnet(data, max_epochs=epochs, seed=rep + 5000)
    h1, h0, g, Sc_lag = nuisances(net, censnet, data)

    psi_naive = naive_plugin(h1, h0)
    psi_os, D_os = one_step(data, h1, h0, g, Sc_lag)
    psi_km = km_by_arm(data)
    psi_ipcw = km_by_arm(data, adjusted=True, g=g, Sc_lag=Sc_lag)
    res = tda(net, censnet, data)
    psi_tda, D = res["psi"], res["D"]

    D_ben, D_rmst = contrast_ifs(D)
    D_ben_os, D_rmst_os = contrast_ifs(D_os)
    crit, _ = simultaneous_band(D_ben, seed=rep)
    out = dict(
        ben_naive=contrast_estimates(psi_naive)[0],
        ben_os=contrast_estimates(psi_os)[0],
        ben_os_iso=contrast_estimates(isotonize(psi_os))[0],
        ben_ipcw=contrast_estimates(psi_ipcw)[0],
        ben_km=contrast_estimates(psi_km)[0],
        ben_tda=contrast_estimates(psi_tda)[0],
        se_ben_tda=if_se(D_ben),
        se_ben_os=if_se(D_ben_os),
        crit_tda=crit,
        rmst_naive=contrast_estimates(psi_naive)[1],
        rmst_os=contrast_estimates(psi_os)[1],
        rmst_tda=contrast_estimates(psi_tda)[1],
        se_rmst_tda=D_rmst.std(ddof=1) / np.sqrt(n),
        se_rmst_os=D_rmst_os.std(ddof=1) / np.sqrt(n),
        n_tda_iters=len(res["history"]),
        tda_converged=float(res.get("converged", np.nan)),
        tda_max_pnd_over_tol=float(np.max(np.abs(res["final_pnd"]) / res["final_tol"]))
        if "final_pnd" in res else np.nan,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--start", type=int, default=0, help="first rep index (for splitting)")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    resdir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(resdir, exist_ok=True)
    path = os.path.join(resdir, f"mc_{args.tag}.npz")
    store = {k: list(v) for k, v in np.load(path).items()} if os.path.exists(path) else {}

    truth = dgp.true_estimands(K=args.K)
    for rep in range(args.start, args.start + args.reps):
        out = run_rep(rep, args.n, args.K, args.epochs)
        for k, v in out.items():
            store.setdefault(k, []).append(v)
        np.savez(path, **{k: np.asarray(v) for k, v in store.items()})
        if (rep - args.start + 1) % 10 == 0 or rep == args.start:
            print(f"rep {rep}: done ({len(store['rmst_tda'])} total stored)")

    # ---- summary ----------------------------------------------------------
    S = {k: np.asarray(v) for k, v in store.items()}
    ben_true = truth["benefit"]
    summary = {
        "benefit_curve": {
            "naive": summarize_curve(S["ben_naive"], ben_true),
            "km_confounded": summarize_curve(S["ben_km"], ben_true),
            "one_step": summarize_curve(S["ben_os"], ben_true, se=S["se_ben_os"]),
            "tda": summarize_curve(S["ben_tda"], ben_true, se=S["se_ben_tda"]),
            "tda_simultaneous": summarize_curve(S["ben_tda"], ben_true,
                                                se=S["se_ben_tda"], crit=S["crit_tda"]),
        },
        "paired_mse_diff_tda_minus_onestep": None,
        "rmst_diff": {
            "naive": summarize_scalar(S["rmst_naive"], truth["rmst_diff"]),
            "one_step": summarize_scalar(S["rmst_os"], truth["rmst_diff"], se=S["se_rmst_os"]),
            "tda": summarize_scalar(S["rmst_tda"], truth["rmst_diff"], se=S["se_rmst_tda"]),
        },
        "mean_tda_iters": float(S["n_tda_iters"].mean()),
        "reps": int(S["rmst_tda"].shape[0]),
    }
    # optional baselines present only in runs made after they were added
    for key, name in [("ben_os_iso", "one_step_isotonized"), ("ben_ipcw", "ipcw_km")]:
        if key in S:
            summary["benefit_curve"][name] = summarize_curve(S[key], ben_true)
    # paired Monte Carlo error on the TDSC - one-step MSE gap (same data/nuisances)
    R = S["ben_tda"].shape[0]
    d = ((S["ben_tda"] - ben_true) ** 2).mean(axis=1) - \
        ((S["ben_os"] - ben_true) ** 2).mean(axis=1)
    summary["paired_mse_diff_tda_minus_onestep"] = dict(
        mean=float(d.mean()), mc_se=float(d.std(ddof=1) / np.sqrt(R)))
    print(json.dumps(summary, indent=2))
    with open(os.path.join(resdir, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
