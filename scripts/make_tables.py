"""Generate LaTeX table rows + a results digest from the bank summaries.

Reads summary_bank_*.json (new projected-criterion banks) and, for causal
survival forests, the legacy summary_comparators.json (CSF depends only on
the data, which is seed-identical, so those rows remain valid).

Writes results/tables_generated.tex (row blocks to paste/inspect) and prints
a digest with every number the results rewrite needs, including safeguard
coverage recomputed from the raw stores.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tdsc import dgp

RES = os.path.join(os.path.dirname(__file__), "..", "results")
Z = 1.959964

LABELS = {
    "naive": "Naive neural plug-in", "km": "Unadjusted KM",
    "ipcw": "IPTW$\\times$IPCW KM", "os": "One-step (AIPCW)",
    "os_iso": "One-step, isotonized", "ostmle_pt": "Output-space TMLE (per-$t$)",
    "ostmle_uni": "Output-space TMLE (universal)",
    "tda": "TDSC plug-in (full-EIF SE)", "tda_projse": "TDSC plug-in (working SE)",
    "tda_topup": "TDSC + full top-up", "cf_plug": "CF plug-in (full-EIF SE)",
    "cf_plug_projse": "CF plug-in (working SE)", "cf_topup": "CF top-up",
}


def row(nm, r):
    cov = f"{r['cov_ptwise']*100:.1f}\\%" if "cov_ptwise" in r else "---"
    wid = f"{r['ci_width']:.3f}" if "ci_width" in r else "---"
    sim = f" & {r['cov_simult']*100:.1f}\\%" if "cov_simult" in r else ""
    return (f"{LABELS.get(nm, nm):32s} & {r['bias']:.4f} & {r['mc_sd']:.4f} & "
            f"{r.get('mean_se', float('nan')):.4f} & {r['mse']:.5f} & {cov} & {wid}{sim} \\\\")


def safeguard(tag, design="linear", n=1000, K=30):
    """Top-up point estimate with the one-step (nonparametric initial-fit) SE."""
    p = os.path.join(RES, f"bank_{tag}.npz")
    if not os.path.exists(p):
        return None
    S = np.load(p)
    truth = dgp.true_estimands(K=K, design=design)["benefit"]
    est, se = S["ben_tda_topup"], S["se_ben_os"]
    return dict(cov=float((np.abs(est - truth) <= Z * se).mean()),
                width=float((2 * Z * se).mean()))


def main():
    out = []
    for p in sorted(glob.glob(os.path.join(RES, "summary_bank_*.json"))):
        d = json.load(open(p))
        tag = os.path.basename(p).replace("summary_bank_", "").replace(".json", "")
        out.append(f"\n%%%%% {tag}  (reps={d['reps']}, scenario={d['scenario']}, "
                   f"design={d['design']}, n={d['n']}, K={d['K']}) %%%%%")
        out.append("% columns: bias & mc_sd & mean_se & mse & cov & width [& simult]")
        for nm in ["naive", "km", "ipcw", "os", "os_iso", "ostmle_pt", "ostmle_uni",
                   "tda", "tda_projse", "tda_topup", "cf_plug", "cf_plug_projse",
                   "cf_topup"]:
            if nm in d["estimators"]:
                out.append(row(nm, d["estimators"][nm]))
        rm = {nm: d["estimators"][nm] for nm in d["estimators"]}
        out.append("% RMST bias / cov: " + "; ".join(
            f"{nm}={r['rmst_bias']:+.3f}" + (f"/{r['rmst_cov']*100:.0f}%" if "rmst_cov" in r else "")
            for nm, r in rm.items() if "rmst_bias" in r))
        out.append(f"% diagnostics: {json.dumps(d['diagnostics'])}")
        design = d["design"]
        sg = safeguard(tag, design=design, n=d["n"], K=d["K"])
        if sg and d["scenario"] == "outcome_mis":
            out.append(f"% safeguard (topup +/- one-step SE): cov {sg['cov']*100:.1f}%, "
                       f"width {sg['width']:.3f}")
    # CSF from the legacy comparator study (data-only, still valid)
    pc = os.path.join(RES, "summary_comparators.json")
    if os.path.exists(pc):
        c = json.load(open(pc))["csf"]["benefit_at_horizons"]
        out.append("\n%%%%% CSF (legacy comparator study; data identical) %%%%%")
        out.append(f"% csf: bias {c['bias']:.4f} mse {c['mse']:.5f} "
                   f"cov {c['coverage_pointwise']*100:.1f}% width {c['ci_width']:.3f}")
    # ridge grid
    for p in sorted(glob.glob(os.path.join(RES, "summary_bank_ridge_*.json"))):
        d = json.load(open(p))
        t = d["estimators"]["tda"]
        out.append(f"% ridge {os.path.basename(p)}: tda mse {t['mse']:.5f} "
                   f"cov {t['cov_ptwise']*100:.1f}% resid "
                   f"{d['diagnostics'].get('tda_proj_resid_rel', float('nan')):.3f}")
    text = "\n".join(out)
    with open(os.path.join(RES, "tables_generated.tex"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
