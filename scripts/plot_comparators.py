"""Comparator summary figure from the unified bank summaries: benefit-curve
MSE and pointwise coverage across estimators and nuisance scenarios, with CSF
(own nuisances; legacy comparator study, data-identical seeds) as reference."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

scenarios = ["well", "outcome_mis", "weights_mis"]
banks = {s: json.load(open(os.path.join(RES, f"summary_bank_{s}_linear_n1000_K30.json")))
         for s in scenarios}
labels = {"well": "well-\nspecified", "outcome_mis": "outcome\nmisspec.",
          "weights_mis": "weights\nmisspec."}
est = [
    ("one-step (AIPCW)", "tab:gray", "o", "os"),
    ("output-space TMLE (universal)", "tab:green", "s", "ostmle_uni"),
    ("TDSC plug-in", "tab:blue", "D", "tda"),
    ("TDSC + top-up", "tab:purple", "^", "tda_topup"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.7))
x = np.arange(len(scenarios))
for i, (name, color, marker, key) in enumerate(est):
    dx = (i - 1.5) * 0.14
    mse = [banks[s]["estimators"][key]["mse"] for s in scenarios]
    cov = [banks[s]["estimators"][key]["cov_ptwise"] * 100 for s in scenarios]
    ax1.plot(x + dx, mse, marker, color=color, ms=7, label=name)
    ax2.plot(x + dx, cov, marker, color=color, ms=7, label=name)

pc = os.path.join(RES, "summary_comparators.json")
if os.path.exists(pc):
    csf = json.load(open(pc))["csf"]["benefit_at_horizons"]
    ax1.axhline(csf["mse"], color="tab:red", ls=":", lw=1.5)
    ax1.text(2.42, csf["mse"] * 1.12, "causal survival forest", color="tab:red",
             fontsize=8, ha="right")
    ax2.axhline(csf["coverage_pointwise"] * 100, color="tab:red", ls=":", lw=1.5)

ax1.set_yscale("log")
ax1.set_ylabel("benefit-curve MSE (log scale)")
ax2.set_ylabel("pointwise coverage (%)")
ax2.axhline(95, color="k", lw=0.8, ls="--")
ax2.set_ylim(25, 100)
for ax in (ax1, ax2):
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in scenarios], fontsize=9)
    ax.grid(alpha=0.25)
ax1.legend(fontsize=8, loc="upper left")
fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, "comparators.png")
fig.savefig(out, dpi=150)
print("saved", out)
