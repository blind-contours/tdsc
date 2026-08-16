"""Safeguarded-variance analysis (no re-simulation needed).

Question: does pairing the targeted point estimate (TDSC, or TDSC + residual
top-up) with the NONPARAMETRIC influence-function SE -- the one-step's SE,
evaluated at the initial fit -- restore coverage where the working-model SE
undercovers (outcome misspecification)?

Reads the stored scenario grids and reports pointwise coverage/width for:
  tda      +/- 1.96 * se_working        (as in the paper's Table 2)
  tda      +/- 1.96 * se_nonparametric
  topup    +/- 1.96 * se_nonparametric  (the recommended safeguard)
and the RMST analogues.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tdsc import dgp

RES = os.path.join(os.path.dirname(__file__), "..", "results")
Z = 1.959964


def cover(est, truth, se):
    return float((np.abs(est - truth) <= Z * se).mean())


def main():
    truth = dgp.true_estimands(K=30)
    out = {}
    for scen in ["well", "outcome_mis", "weights_mis"]:
        S = dict(np.load(os.path.join(RES, f"mc_scen_{scen}.npz")))
        bt, rt = truth["benefit"], truth["rmst_diff"]
        rows = {
            "tda_working_se": (S["ben_tda"], S["se_ben_tda"], S["rmst_tda"], S["se_rmst_tda"]),
            "tda_nonpar_se": (S["ben_tda"], S["se_ben_os"], S["rmst_tda"], S["se_rmst_os"]),
            "topup_nonpar_se": (S["ben_tda_topup"], S["se_ben_os"],
                                S["rmst_tda_topup"], S["se_rmst_os"]),
            "onestep_nonpar_se": (S["ben_os"], S["se_ben_os"], S["rmst_os"], S["se_rmst_os"]),
        }
        out[scen] = {
            name: dict(cov_ptwise=cover(b, bt, sb), ci_width=float((2 * Z * sb).mean()),
                       rmst_cov=cover(r, rt, sr), rmst_width=float((2 * Z * sr).mean()))
            for name, (b, sb, r, sr) in rows.items()
        }
    print(json.dumps(out, indent=1))
    with open(os.path.join(RES, "summary_safeguard.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
