"""Unified paired simulation bank (post-review architecture).

One configuration = (scenario, design, n, K). All estimators run on identical
data and, where applicable, identical nuisance fits (standalone propensity
network throughout so nuisance arms can be blinded independently):

  naive, unadjusted KM, IPTW-IPCW KM, one-step (+isotonized, +simultaneous
  band), per-timepoint output-space TMLE, UNIVERSAL output-space TMLE,
  TDSC (projected-EIF stopping rule; both full-EIF and projected-EIF SEs),
  TDSC + full-residual top-up, and (with --cf) cross-fitted plug-in and
  top-up with pooled full and projected IFs.

Summaries report bias / MC-SD / mean estimated SE / MSE / coverage / width
so variance-estimation behavior is diagnosable.

Usage examples:
  python scripts/run_bank.py --scenario well --design linear --n 1000 --reps 150 --cf
  python scripts/run_bank.py --scenario well --design nonlinear --n 1000 --reps 150
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
from tdsc.estimators import isotonize, km_by_arm, naive_plugin, one_step, tda
from tdsc.influence import contrast_estimates, contrast_ifs
from tdsc.model import nuisances, train_censnet, train_dragonsurv, train_propnet
from tdsc.ostmle import ostmle_curve, universal_ostmle

SCENARIOS = {
    "well": dict(cols_out=None, cols_w=None),
    "outcome_mis": dict(cols_out=[1, 3, 4, 5, 6, 7, 8, 9], cols_w=None),
    "weights_mis": dict(cols_out=None, cols_w=[2, 3, 4, 5, 6, 7, 8, 9]),
}


def curve_entry(out, name, psi, D=None, D_for_se=None, band_seed=None):
    ben, rmst = contrast_estimates(psi)
    out[f"ben_{name}"] = ben
    out[f"rmst_{name}"] = rmst
    Dse = D_for_se if D_for_se is not None else D
    if Dse is not None:
        D_ben, D_rmst = contrast_ifs(Dse)
        out[f"se_ben_{name}"] = if_se(D_ben)
        out[f"se_rmst_{name}"] = D_rmst.std(ddof=1) / np.sqrt(D_ben.shape[0])
        if band_seed is not None:
            out[f"crit_{name}"], _ = simultaneous_band(D_ben, seed=band_seed)


def run_rep(rep, args, cfg):
    data = dgp.simulate(args.n, K=args.K, seed=1000 + rep, design=args.design)
    net = train_dragonsurv(data, max_epochs=args.epochs, seed=rep, cols=cfg["cols_out"])
    censnet = train_censnet(data, max_epochs=args.epochs, seed=rep + 5000,
                            cols=cfg["cols_w"])
    propnet = train_propnet(data, max_epochs=args.epochs, seed=rep + 9000,
                            cols=cfg["cols_w"])
    h1, h0, g, Sc = nuisances(net, censnet, data, propnet=propnet)

    out = {}
    curve_entry(out, "naive", naive_plugin(h1, h0))
    curve_entry(out, "km", km_by_arm(data))
    curve_entry(out, "ipcw", km_by_arm(data, adjusted=True, g=g, Sc_lag=Sc))
    psi_os, D_os = one_step(data, h1, h0, g, Sc)
    curve_entry(out, "os", psi_os, D=D_os, band_seed=rep)
    curve_entry(out, "os_iso", isotonize(psi_os))
    psi_pt, D_pt = ostmle_curve(data, h1, h0, g, Sc)
    curve_entry(out, "ostmle_pt", psi_pt, D=D_pt)
    psi_uni, D_uni, uni_iters = universal_ostmle(data, h1, h0, g, Sc)
    curve_entry(out, "ostmle_uni", psi_uni, D=D_uni, band_seed=rep + 1)
    out["uni_iters"] = float(uni_iters)

    res = tda(net, censnet, data, propnet=propnet, ridge=args.ridge)
    curve_entry(out, "tda", res["psi"], D=res["D"], band_seed=rep + 2)
    curve_entry(out, "tda_projse", res["psi"], D_for_se=res["D_proj"],
                band_seed=rep + 3)
    curve_entry(out, "tda_topup", res["psi"] + res["final_pnd"], D=res["D"])
    out["tda_converged_proj"] = float(res["converged"])
    out["tda_iters"] = float(len(res["history"]))
    out["tda_proj_resid_rel"] = float(res["history"][-1].get("proj_resid_rel", np.nan))
    out["tda_max_pnd_full"] = float(np.max(np.abs(res["final_pnd"])))

    if args.cf:
        cf = tda_crossfit(data, V=args.V, seed=rep, epochs=args.epochs,
                          cols_out=cfg["cols_out"], cols_w=cfg["cols_w"])
        curve_entry(out, "cf_topup", cf["psi"], D=cf["D"], band_seed=rep + 4)
        curve_entry(out, "cf_plug", cf["psi_plug"], D=cf["D"])
        curve_entry(out, "cf_plug_projse", cf["psi_plug"], D_for_se=cf["D_proj"])
    return out


def summarize(store, truth, args):
    S = {k: np.asarray(v) for k, v in store.items()}
    bt, rt = truth["benefit"], truth["rmst_diff"]
    names = sorted({k[4:] for k in S if k.startswith("ben_")})
    rows = {}
    for nm in names:
        est = S[f"ben_{nm}"]
        row = dict(bias=float(np.abs(est.mean(axis=0) - bt).mean()),
                   mc_sd=float(est.std(axis=0, ddof=1).mean()),
                   mse=float(((est - bt) ** 2).mean()))
        if f"se_ben_{nm}" in S:
            se = S[f"se_ben_{nm}"]
            row["mean_se"] = float(se.mean())
            row["cov_ptwise"] = float((np.abs(est - bt) <= 1.959964 * se).mean())
            row["ci_width"] = float((2 * 1.959964 * se).mean())
            if f"crit_{nm}" in S:
                crit = S[f"crit_{nm}"][:, None]
                row["cov_simult"] = float(
                    (np.abs(est - bt) <= crit * se).all(axis=1).mean())
        rmst = S[f"rmst_{nm}"]
        row["rmst_bias"] = float(rmst.mean() - rt)
        if f"se_rmst_{nm}" in S:
            row["rmst_cov"] = float(
                (np.abs(rmst - rt) <= 1.959964 * S[f"se_rmst_{nm}"]).mean())
        rows[nm] = row
    diag = {k: float(S[k].mean()) for k in
            ["tda_converged_proj", "tda_iters", "tda_proj_resid_rel",
             "tda_max_pnd_full", "uni_iters"] if k in S}
    return dict(scenario=args.scenario, design=args.design, n=args.n, K=args.K,
                reps=int(S[f"ben_{names[0]}"].shape[0]), estimators=rows,
                diagnostics=diag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="well")
    ap.add_argument("--design", choices=["linear", "nonlinear", "nearpos"],
                    default="linear")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--cf", action="store_true")
    ap.add_argument("--V", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=1e-2,
                    help="relative ridge for the targeting projection")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cfg = SCENARIOS[args.scenario]
    tag = args.tag or f"{args.scenario}_{args.design}_n{args.n}_K{args.K}"

    resdir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(resdir, exist_ok=True)
    path = os.path.join(resdir, f"bank_{tag}.npz")
    store = {k: list(v) for k, v in np.load(path).items()} if os.path.exists(path) else {}

    truth = dgp.true_estimands(K=args.K, design=args.design)
    for rep in range(args.start, args.start + args.reps):
        out = run_rep(rep, args, cfg)
        for k, v in out.items():
            store.setdefault(k, []).append(v)
        np.savez(path, **{k: np.asarray(v) for k, v in store.items()})
        if (rep - args.start + 1) % 10 == 0 or rep == args.start:
            print(f"[{tag}] rep {rep} done ({len(store['rmst_tda'])} stored)",
                  flush=True)

    summary = summarize(store, truth, args)
    print(json.dumps(summary, indent=2))
    with open(os.path.join(resdir, f"summary_bank_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
