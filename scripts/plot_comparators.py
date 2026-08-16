"""Comparator summary figure: benefit-curve MSE and pointwise coverage across
estimators and nuisance scenarios (plus CSF, which fits its own nuisances)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")

scen_j = {s: json.load(open(os.path.join(RES, f"summary_scen_{s}.json")))
          for s in ["well", "outcome_mis", "weights_mis"]}
comp_j = json.load(open(os.path.join(RES, "summary_comparators.json")))

scenarios = ["well", "outcome_mis", "weights_mis"]
labels = {"well": "well-\nspecified", "outcome_mis": "outcome\nmisspec.",
          "weights_mis": "weights\nmisspec."}
est = [
    ("one-step (AIPCW)", "tab:gray", "o",
     lambda s: scen_j[s]["benefit_curve"]["one_step"]),
    ("output-space TMLE", "tab:green", "s",
     lambda s: comp_j["ostmle"][s]["benefit_curve"]),
    ("TDSC (weight-space)", "tab:blue", "D",
     lambda s: scen_j[s]["benefit_curve"]["tda"]),
]
csf = comp_j["csf"]["benefit_at_horizons"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
x = np.arange(len(scenarios))
for i, (name, color, marker, get) in enumerate(est):
    dx = (i - 1) * 0.16
    mse = [get(s)["mse"] for s in scenarios]
    cov = [get(s)["coverage_pointwise"] * 100 for s in scenarios]
    ax1.plot(x + dx, mse, marker, color=color, ms=7, label=name)
    ax2.plot(x + dx, cov, marker, color=color, ms=7, label=name)
# CSF: own nuisances, scenario-independent -> horizontal reference
ax1.axhline(csf["mse"], color="tab:red", ls=":", lw=1.5)
ax1.text(2.42, csf["mse"] * 1.12, "causal survival forest", color="tab:red",
         fontsize=8, ha="right")
ax2.axhline(csf["coverage_pointwise"] * 100, color="tab:red", ls=":", lw=1.5)

ax1.set_yscale("log")
ax1.set_ylabel("benefit-curve MSE (log scale)")
ax2.set_ylabel("pointwise coverage (%)")
ax2.axhline(95, color="k", lw=0.8, ls="--")
ax2.set_ylim(60, 100)
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
