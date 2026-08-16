"""External-comparator Monte Carlo: output-space survival TMLE + causal
survival forests, head-to-head with the stored TDSC / one-step results.

- osTMLE uses the SAME nuisances (network seeds) as scripts/run_scenarios.py,
  so per-replication results pair exactly with results/mc_scen_*.npz across
  {well, outcome_mis, weights_mis}. This completes the mechanism decomposition:
  one-step (output-space additive) vs osTMLE (output-space TMLE) vs TDSC
  (weight-space TMLE), all on identical nuisance fits.
- Causal survival forests (grf, via scripts/csf_bridge.R) fit their own
  nuisances, so they run once per replication (the raw data are identical
  across scenarios) at horizons 5,10,15,20,25.

Usage: python scripts/run_comparators.py --reps 100
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tdsc import dgp
from tdsc.bands import if_se
from tdsc.influence import contrast_estimates, contrast_ifs
from tdsc.metrics import summarize_curve, summarize_scalar
from tdsc.model import nuisances, train_censnet, train_dragonsurv, train_propnet
from tdsc.ostmle import ostmle_curve

SCENARIOS = {
    "well": dict(cols_out=None, cols_w=None),
    "outcome_mis": dict(cols_out=[1, 3, 4, 5, 6, 7, 8, 9], cols_w=None),
    "weights_mis": dict(cols_out=None, cols_w=[2, 3, 4, 5, 6, 7, 8, 9]),
}
CSF_HORIZONS = [5, 10, 15, 20, 25]
BRIDGE = os.path.join(os.path.dirname(__file__), "csf_bridge.R")


def run_csf(data, num_trees=1000):
    with tempfile.TemporaryDirectory() as td:
        inp, outp = os.path.join(td, "d.csv"), os.path.join(td, "o.csv")
        with open(inp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"X{j+1}" for j in range(data["X"].shape[1])]
                       + ["A", "Ttil", "Delta"])
            for i in range(len(data["A"])):
                w.writerow(list(data["X"][i])
                           + [data["A"][i], data["Ttil"][i], data["Delta"][i]])
        subprocess.run(["Rscript", BRIDGE, inp, outp,
                        ",".join(map(str, CSF_HORIZONS)), str(num_trees)],
                       check=True, capture_output=True)
        rows = list(csv.DictReader(open(outp)))
    est = np.array([float(r["est"]) for r in rows])
    se = np.array([float(r["se"]) for r in rows])
    return est, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    resdir = os.path.join(os.path.dirname(__file__), "..", "results")
    path = os.path.join(resdir, "mc_comparators.npz")
    store = {k: list(v) for k, v in np.load(path).items()} if os.path.exists(path) else {}

    truth = dgp.true_estimands(K=args.K)
    for rep in range(args.start, args.start + args.reps):
        data = dgp.simulate(args.n, K=args.K, seed=1000 + rep)
        out = {}
        for scen, cfg in SCENARIOS.items():
            net = train_dragonsurv(data, max_epochs=args.epochs, seed=rep,
                                   cols=cfg["cols_out"])
            censnet = train_censnet(data, max_epochs=args.epochs, seed=rep + 5000,
                                    cols=cfg["cols_w"])
            propnet = train_propnet(data, max_epochs=args.epochs, seed=rep + 9000,
                                    cols=cfg["cols_w"])
            h1, h0, g, Sc = nuisances(net, censnet, data, propnet=propnet)
            psi, D = ostmle_curve(data, h1, h0, g, Sc)
            D_ben, D_rmst = contrast_ifs(D)
            out[f"ben_ostmle_{scen}"] = contrast_estimates(psi)[0]
            out[f"se_ben_ostmle_{scen}"] = if_se(D_ben)
            out[f"rmst_ostmle_{scen}"] = contrast_estimates(psi)[1]
            out[f"se_rmst_ostmle_{scen}"] = D_rmst.std(ddof=1) / np.sqrt(args.n)
        est, se = run_csf(data)
        out["csf_est"], out["csf_se"] = est, se
        for k, v in out.items():
            store.setdefault(k, []).append(v)
        np.savez(path, **{k: np.asarray(v) for k, v in store.items()})
        if (rep - args.start + 1) % 10 == 0 or rep == args.start:
            print(f"comparators rep {rep}: done ({len(store['csf_est'])} stored)",
                  flush=True)

    S = {k: np.asarray(v) for k, v in store.items()}
    ben_true = truth["benefit"]
    summary = {"reps": int(S["csf_est"].shape[0]), "ostmle": {}, "csf": {}}
    for scen in SCENARIOS:
        summary["ostmle"][scen] = {
            "benefit_curve": summarize_curve(S[f"ben_ostmle_{scen}"], ben_true,
                                             se=S[f"se_ben_ostmle_{scen}"]),
            "rmst": summarize_scalar(S[f"rmst_ostmle_{scen}"], truth["rmst_diff"],
                                     se=S[f"se_rmst_ostmle_{scen}"]),
        }
    keep = [i for i, h in enumerate(CSF_HORIZONS) if h <= args.K]
    csf_true = ben_true[[CSF_HORIZONS[i] - 1 for i in keep]]
    summary["csf"] = {
        "horizons": [CSF_HORIZONS[i] for i in keep],
        "benefit_at_horizons": summarize_curve(S["csf_est"][:, keep], csf_true,
                                               se=S["csf_se"][:, keep]),
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(resdir, "summary_comparators.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
