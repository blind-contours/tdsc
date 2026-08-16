"""Monte Carlo for cross-fitted TDSC, paired against the stored no-split run.

Data seeds (1000 + rep) match scripts/run_montecarlo.py, so replications pair
one-to-one with results/mc_pilot.npz for a direct no-split vs cross-fit
comparison on identical datasets.

Usage: python scripts/run_crossfit.py --reps 100 [--V 5]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tdsc import dgp
from tdsc.bands import if_se, simultaneous_band
from tdsc.crossfit import tda_crossfit
from tdsc.influence import contrast_estimates, contrast_ifs
from tdsc.metrics import summarize_curve, summarize_scalar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--V", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    resdir = os.path.join(os.path.dirname(__file__), "..", "results")
    path = os.path.join(resdir, "mc_crossfit.npz")
    store = {k: list(v) for k, v in np.load(path).items()} if os.path.exists(path) else {}

    truth = dgp.true_estimands(K=args.K)
    for rep in range(args.start, args.start + args.reps):
        data = dgp.simulate(args.n, K=args.K, seed=1000 + rep)
        res = tda_crossfit(data, V=args.V, seed=rep, epochs=args.epochs)
        D_ben, D_rmst = contrast_ifs(res["D"])
        crit, _ = simultaneous_band(D_ben, seed=rep)
        out = dict(
            ben_cf=contrast_estimates(res["psi"])[0],
            ben_cf_topup=contrast_estimates(res["psi_topup"])[0],
            se_ben_cf=if_se(D_ben),
            crit_cf=crit,
            rmst_cf=contrast_estimates(res["psi"])[1],
            rmst_cf_topup=contrast_estimates(res["psi_topup"])[1],
            se_rmst_cf=D_rmst.std(ddof=1) / np.sqrt(args.n),
            mean_fold_unsolved=float(np.mean([d["unsolved"] for d in res["diags"]])),
        )
        for k, v in out.items():
            store.setdefault(k, []).append(v)
        np.savez(path, **{k: np.asarray(v) for k, v in store.items()})
        if (rep - args.start + 1) % 10 == 0 or rep == args.start:
            print(f"cf rep {rep}: done ({len(store['rmst_cf'])} stored)", flush=True)

    S = {k: np.asarray(v) for k, v in store.items()}
    ben_true = truth["benefit"]
    R = S["ben_cf"].shape[0]
    summary = {
        "benefit_curve": {
            "crossfit": summarize_curve(S["ben_cf"], ben_true, se=S["se_ben_cf"]),
            "crossfit_topup": summarize_curve(S["ben_cf_topup"], ben_true, se=S["se_ben_cf"]),
            "crossfit_simultaneous": summarize_curve(S["ben_cf"], ben_true,
                                                     se=S["se_ben_cf"], crit=S["crit_cf"]),
        },
        "rmst_diff": {
            "crossfit": summarize_scalar(S["rmst_cf"], truth["rmst_diff"], se=S["se_rmst_cf"]),
            "crossfit_topup": summarize_scalar(S["rmst_cf_topup"], truth["rmst_diff"],
                                               se=S["se_rmst_cf"]),
        },
        "mean_fold_unsolved": float(S["mean_fold_unsolved"].mean()),
        "reps": int(R),
    }
    # paired comparison vs the stored no-split run on shared data seeds
    pilot_path = os.path.join(resdir, "mc_pilot.npz")
    if os.path.exists(pilot_path):
        P = np.load(pilot_path)
        m = min(R, P["ben_tda"].shape[0])
        d = ((S["ben_cf"][:m] - ben_true) ** 2).mean(axis=1) - \
            ((P["ben_tda"][:m] - ben_true) ** 2).mean(axis=1)
        summary["paired_mse_diff_cf_minus_nosplit"] = dict(
            mean=float(d.mean()), mc_se=float(d.std(ddof=1) / np.sqrt(m)), pairs=int(m))
    print(json.dumps(summary, indent=2))
    with open(os.path.join(resdir, "summary_crossfit.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
