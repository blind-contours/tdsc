# Targeted Deep Survival Contrasts (TDSC)

Valid inference for individualized treatment benefit with neural networks:
extends **Targeted Deep Architectures** (TDA, [arXiv:2507.12435](https://arxiv.org/abs/2507.12435))
from the marginal survival curve to **treatment-specific counterfactual survival
curves and their contrast** — S₁(t), S₀(t), the benefit curve S₁(t)−S₀(t), and
the RMST difference — under **confounding and covariate-dependent censoring**,
with one universal targeting update and multiplier-bootstrap **simultaneous
confidence bands** on the benefit curve.

Companion code for the paper draft in `paper/` (McCoy, Li, van der Laan).

## Method

Discrete-time survival on K bins. A DragonNet-style network (shared trunk,
propensity head, two hazard heads) is trained on the observed-arm NLL; a
separate network fits the censoring hazard. The TDA targeting submodel is the
final linear layer of the two outcome heads (p = 2·K·(hidden+1) parameters).
Each iteration ridge-projects the 2K stacked discrete-time EIFs
(Moore & van der Laan form, with IPTW × IPCW weights) onto per-sample loss
gradients — closed-form for the final layer — merges the directions with the
universal weights of TDA §2.3, and line-searches a step until every
|Pₙ Dₖ| ≤ sd(Dₖ)/(√n log n) or no step improves. Diagnostics (relative
projection residual, unsolved-coordinate counts) and a one-step residual
top-up for unconverged coordinates are built in.

## Layout

```
tdsc/dgp.py         confounded, censored discrete-time DGP + exact truth
tdsc/model.py       DragonSurv + CensNet + PropNet, training, nuisance extraction
tdsc/influence.py   stacked EIFs for S_a(t); benefit-curve and RMST contrasts
tdsc/targeting.py   universal TDA targeting loop (closed-form last-layer gradients)
tdsc/estimators.py  naive plug-in, one-step AIPCW (+isotonized), (IPCW-)KM, TDSC
tdsc/bands.py       IF-based SEs, multiplier-bootstrap sup-t bands
tdsc/metrics.py     Monte Carlo bias/variance/MSE/coverage aggregation
scripts/run_single.py      one dataset + the benefit-curve figure
scripts/run_montecarlo.py  main MC study (resumable; results/mc_pilot.npz)
scripts/run_scenarios.py   nuisance-misspecification grid (well/outcome_mis/weights_mis)
paper/main.tex             manuscript source
results/                   raw MC stores (.npz) + summaries (.json) reported in the paper
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/run_single.py --n 1000 --K 30          # demo + figures/benefit_curve.png
python scripts/run_montecarlo.py --reps 200           # Table 1 (appendable via --start/--tag)
python scripts/run_scenarios.py --scenario well --reps 150         # Table 2 rows
python scripts/run_scenarios.py --scenario outcome_mis --reps 150
python scripts/run_scenarios.py --scenario weights_mis --reps 150
```

All runs checkpoint per replication and are resumable. The `results/` directory
contains the exact stores behind the paper's tables.

## Status / roadmap

- [x] Universal targeting for S₁(·), S₀(·), benefit curve, RMST difference
- [x] Simultaneous sup-t bands (multiplier bootstrap)
- [x] 200-rep main study; 3×150-rep misspecification grid
- [x] Mechanism comparators: isotonized one-step, IPCW-KM
- [x] Convergence diagnostics + one-step residual top-up
- [x] n-scaling (500/1000/2000) + safeguarded-variance analysis
- [ ] Cross-fitted variant
- [ ] Output-space survival TMLE and causal survival forest comparators
- [ ] Calibration-within-predicted-benefit-strata targets
- [ ] Frozen-trunk (foundation-model embedding) experiment
