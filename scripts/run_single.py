"""Single-dataset demo: fit, target, and draw the money plot.

Usage: python scripts/run_single.py [--n 1000] [--K 30] [--seed 7]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tdsc import dgp
from tdsc.bands import if_se, simultaneous_band
from tdsc.estimators import km_by_arm, naive_plugin, one_step, tda
from tdsc.influence import contrast_estimates, contrast_ifs
from tdsc.model import nuisances, train_censnet, train_dragonsurv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "figures"))
    args = ap.parse_args()

    print(f"Simulating n={args.n}, K={args.K} ...")
    data = dgp.simulate(args.n, K=args.K, seed=args.seed)
    print(f"  events: {data['Delta'].mean():.1%}, treated: {data['A'].mean():.1%}")
    truth = dgp.true_estimands(K=args.K)

    print("Training DragonSurv + CensNet ...")
    net = train_dragonsurv(data, max_epochs=args.epochs, seed=args.seed)
    censnet = train_censnet(data, max_epochs=args.epochs, seed=args.seed + 1)

    h1, h0, g, Sc_lag = nuisances(net, censnet, data)
    psi_naive = naive_plugin(h1, h0)
    psi_os, _ = one_step(data, h1, h0, g, Sc_lag)
    psi_km = km_by_arm(data)
    psi_ipcw = km_by_arm(data, adjusted=True, g=g, Sc_lag=Sc_lag)

    print("TDA universal targeting ...")
    res = tda(net, censnet, data, verbose=True)
    psi_tda, D = res["psi"], res["D"]

    ben_true = truth["benefit"]
    ben = {name: contrast_estimates(p)[0] for name, p in
           [("naive", psi_naive), ("one-step", psi_os), ("km", psi_km),
            ("ipcw-km", psi_ipcw), ("tda", psi_tda)]}
    D_ben, D_rmst = contrast_ifs(D)
    se_ben = if_se(D_ben)
    crit, _ = simultaneous_band(D_ben, seed=args.seed)

    rmst_est = contrast_estimates(psi_tda)[1]
    rmst_se = D_rmst.std(ddof=1) / np.sqrt(args.n)
    print(f"\nRMST difference: truth {truth['rmst_diff']:.3f} | "
          f"TDA {rmst_est:.3f} +/- {1.96 * rmst_se:.3f} | "
          f"naive {contrast_estimates(psi_naive)[1]:.3f} | "
          f"KM (confounded) {contrast_estimates(psi_km)[1]:.3f}")
    print(f"Simultaneous band critical value: {crit:.3f} (vs 1.96 pointwise)")
    mae = {k: float(np.abs(v - ben_true).mean()) for k, v in ben.items()}
    print("Benefit-curve MAE:", {k: round(v, 4) for k, v in sorted(mae.items())})

    # ---- money plot -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(1, args.K + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(t, ben["tda"] - crit * se_ben, ben["tda"] + crit * se_ben,
                    alpha=0.18, color="tab:blue", label="TDSC simultaneous 95% band")
    ax.fill_between(t, ben["tda"] - 1.96 * se_ben, ben["tda"] + 1.96 * se_ben,
                    alpha=0.30, color="tab:blue", label="TDSC pointwise 95% CI")
    ax.plot(t, ben_true, "k-", lw=2, label="Truth $S_1(t)-S_0(t)$")
    ax.plot(t, ben["tda"], color="tab:blue", lw=2, label="TDSC")
    ax.plot(t, ben["naive"], color="tab:orange", ls="--", lw=1.5, label="Naive neural plug-in")
    ax.plot(t, ben["km"], color="tab:red", ls=":", lw=1.5, label="Unadjusted KM (confounded)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("time bin")
    ax.set_ylabel("treatment benefit on survival")
    ax.set_title("Counterfactual survival benefit curve: TDSC vs baselines")
    ax.legend(loc="best", fontsize=9)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "benefit_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
