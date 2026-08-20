"""Two paper figures built from the raw unified-bank Monte Carlo stores.

1. figures/grid_calibration.png -- where the estimators live on the time grid:
   per-timepoint bias, SE calibration (MC SD vs mean estimated SE), pointwise
   coverage across t, and n x MSE across sample sizes.
2. figures/misspec_story.png -- the two-estimator story: sampling distributions
   of the RMST difference under outcome misspecification (plug-in tracks the
   working parameter, top-up recovers the truth) and the in-sample diagnostics
   that separate the two regimes replication by replication.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tdsc import dgp

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

K = 30
truth = dgp.true_estimands(K=K, design="linear")
ben0, rmst0 = truth["benefit"], truth["rmst_diff"]

EST = [
    ("one-step (AIPCW)", "tab:gray", "os"),
    ("output-space TMLE (univ.)", "tab:green", "ostmle_uni"),
    ("TDSC plug-in", "tab:blue", "tda"),
    ("TDSC + top-up", "tab:purple", "tda_topup"),
]


def load(name):
    return np.load(os.path.join(RES, name), allow_pickle=True)


def cov_curve(bank, key, se_key=None):
    est = bank[f"ben_{key}"]
    se = bank[se_key if se_key else f"se_ben_{key}"]
    hit = np.abs(est - ben0[None, :]) <= 1.96 * se
    return hit.mean(axis=0)


# ---------------------------------------------------------------- figure 1
well = load("bank_well_linear_n1000_K30.npz")
t = np.arange(1, K + 1)

fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
(ax_b, ax_se), (ax_c, ax_n) = axes

for name, color, key in EST:
    bias = np.abs(well[f"ben_{key}"].mean(axis=0) - ben0)
    ax_b.plot(t, bias, color=color, lw=1.6, label=name)
ax_b.set_ylabel(r"$|$Monte Carlo bias$|$ of $\hat\beta(t)$")
ax_b.set_title("(a) per-timepoint bias, well-specified $n{=}1000$", fontsize=10)
ax_b.legend(fontsize=8)

ax_se.plot(t, well["ben_os"].std(axis=0), color="tab:gray", lw=1.6,
           label="one-step: MC SD")
ax_se.plot(t, well["se_ben_os"].mean(axis=0), color="tab:gray", lw=1.4, ls="--",
           label="one-step: mean est. SE")
ax_se.plot(t, well["ben_tda"].std(axis=0), color="tab:blue", lw=1.6,
           label="TDSC plug-in: MC SD")
ax_se.plot(t, well["se_ben_tda"].mean(axis=0), color="tab:blue", lw=1.4, ls="--",
           label="full-EIF SE")
ax_se.plot(t, well["se_ben_tda_projse"].mean(axis=0), color="tab:cyan", lw=1.4,
           ls=":", label="working-model SE")
ax_se.set_ylabel("SD / SE of $\\hat\\beta(t)$")
ax_se.set_title("(b) sampling SD vs. estimated SE", fontsize=10)
ax_se.legend(fontsize=8)

mc_band = 1.96 * np.sqrt(0.95 * 0.05 / well["ben_os"].shape[0]) * 100
ax_c.axhspan(95 - mc_band, 95 + mc_band, color="k", alpha=0.08,
             label="95% $\\pm$ MC error")
ax_c.axhline(95, color="k", lw=0.8, ls="--")
for name, color, key in EST:
    ax_c.plot(t, 100 * cov_curve(well, key), color=color, lw=1.6, label=name)
ax_c.set_ylim(80, 100)
ax_c.set_xlabel("time bin $t$")
ax_c.set_ylabel("pointwise coverage (%)")
ax_c.set_title("(c) coverage across the grid", fontsize=10)
ax_c.legend(fontsize=8, loc="lower left", ncol=2)

ns = [500, 1000, 2000]
for name, color, key in [e for e in EST if e[2] != "ostmle_uni"]:
    nmse = []
    for n in ns:
        b = load(f"bank_well_linear_n{n}_K30.npz")
        nmse.append(n * ((b[f"ben_{key}"] - ben0[None, :]) ** 2).mean())
    ax_n.plot(ns, nmse, "o-", color=color, lw=1.6, ms=6, label=name)
ax_n.set_xscale("log")
ax_n.set_xticks(ns)
ax_n.set_xticklabels([str(n) for n in ns])
ax_n.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax_n.set_xlabel("sample size $n$")
ax_n.set_ylabel(r"$n \times$ benefit-curve MSE")
ax_n.set_ylim(bottom=0)
ax_n.set_title("(d) $n\\times$MSE across sample sizes", fontsize=10)
ax_n.legend(fontsize=8)

for ax in axes.flat:
    ax.grid(alpha=0.25)
fig.tight_layout()
out = os.path.join(FIG, "grid_calibration.png")
fig.savefig(out, dpi=150)
print("saved", out)

# ---------------------------------------------------------------- figure 2
mis = load("bank_outcome_mis_linear_n1000_K30.npz")

fig, (ax_v, ax_d) = plt.subplots(1, 2, figsize=(10.2, 3.9))

dist = [
    ("naive\nplug-in", "tab:orange", mis["rmst_naive"]),
    ("one-step\n(AIPCW)", "tab:gray", mis["rmst_os"]),
    ("TDSC\nplug-in", "tab:blue", mis["rmst_tda"]),
    ("TDSC\n+ top-up", "tab:purple", mis["rmst_tda_topup"]),
    ("CF\ntop-up", "tab:brown", mis["rmst_cf_topup"]),
]
parts = ax_v.violinplot([d[2] for d in dist], showmedians=True, widths=0.7)
for body, (_, color, _) in zip(parts["bodies"], dist):
    body.set_facecolor(color)
    body.set_alpha(0.55)
for k in ("cmedians", "cmins", "cmaxes", "cbars"):
    parts[k].set_color("k")
    parts[k].set_linewidth(0.9)
ax_v.axhline(rmst0, color="tab:red", lw=1.4, ls="--",
             label=f"truth $\\Delta_{{\\mathrm{{RMST}}}}={rmst0:.2f}$")
ax_v.set_xticks(range(1, len(dist) + 1))
ax_v.set_xticklabels([d[0] for d in dist], fontsize=8)
ax_v.set_ylabel(r"$\widehat\Delta_{\mathrm{RMST}}$ across replications")
ax_v.set_title("(a) outcome misspecification: sampling distributions",
               fontsize=10)
ax_v.legend(fontsize=8)
ax_v.grid(alpha=0.25, axis="y")

ax_d.scatter(well["tda_proj_resid_rel"], well["tda_max_pnd_full"], s=16,
             color="tab:blue", alpha=0.6, label="well-specified")
ax_d.scatter(mis["tda_proj_resid_rel"], mis["tda_max_pnd_full"], s=16,
             color="tab:red", alpha=0.6, marker="^",
             label="outcome misspecified")
ax_d.set_xlabel(r"projection residual $\Vert D-\widehat D^{w,\lambda}\Vert/\Vert D\Vert$")
ax_d.set_ylabel(r"max full-EIF residual $\max_j |P_n D_j|$")
ax_d.set_yscale("log")
ax_d.set_title("(b) in-sample diagnostics separate the regimes", fontsize=10)
ax_d.legend(fontsize=8)
ax_d.grid(alpha=0.25)

fig.tight_layout()
out = os.path.join(FIG, "misspec_story.png")
fig.savefig(out, dpi=150)
print("saved", out)

print("truth rmst:", rmst0)
print("median rmst plug-in (mis):", np.median(mis["rmst_tda"]))
print("median rmst top-up (mis):", np.median(mis["rmst_tda_topup"]))
print("diag means well:", well["tda_proj_resid_rel"].mean(),
      well["tda_max_pnd_full"].mean())
print("diag means mis:", mis["tda_proj_resid_rel"].mean(),
      mis["tda_max_pnd_full"].mean())
