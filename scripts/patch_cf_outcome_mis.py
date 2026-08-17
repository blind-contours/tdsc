"""Recompute the cross-fit columns of the outcome_mis bank with correct
scenario blinding (the original run trained unblinded fold nets -- see
crossfit.py history). Overwrites only the cf_* keys, preserving rep order,
then regenerates the summary."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tdsc import dgp
from tdsc.bands import if_se, simultaneous_band
from tdsc.crossfit import tda_crossfit
from tdsc.influence import contrast_estimates, contrast_ifs

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PATH = os.path.join(RES, "bank_outcome_mis_linear_n1000_K30.npz")
COLS_OUT = [1, 3, 4, 5, 6, 7, 8, 9]

store = {k: list(v) for k, v in np.load(PATH).items()}
reps = len(store["rmst_tda"])
for rep in range(reps):
    data = dgp.simulate(1000, K=30, seed=1000 + rep)
    cf = tda_crossfit(data, V=5, seed=rep, epochs=500,
                      cols_out=COLS_OUT, cols_w=None)
    for name, psi, Dse in [("cf_topup", cf["psi"], cf["D"]),
                           ("cf_plug", cf["psi_plug"], cf["D"]),
                           ("cf_plug_projse", cf["psi_plug"], cf["D_proj"])]:
        ben, rmst = contrast_estimates(psi)
        store[f"ben_{name}"][rep] = ben
        store[f"rmst_{name}"][rep] = rmst
        D_ben, D_rmst = contrast_ifs(Dse)
        store[f"se_ben_{name}"][rep] = if_se(D_ben)
        store[f"se_rmst_{name}"][rep] = D_rmst.std(ddof=1) / np.sqrt(1000)
        if name == "cf_topup":
            crit, _ = simultaneous_band(D_ben, seed=rep + 4)
            store[f"crit_{name}"][rep] = crit
    np.savez(PATH, **{k: np.asarray(v) for k, v in store.items()})
    if (rep + 1) % 10 == 0 or rep == 0:
        print(f"cf-patch rep {rep} done", flush=True)
print("patch complete")
