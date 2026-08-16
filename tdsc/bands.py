"""IF-based standard errors and multiplier-bootstrap simultaneous bands."""
import numpy as np


def if_se(D):
    """Per-target SE from the IF matrix: sd(D_k)/sqrt(n)."""
    n = D.shape[0]
    return D.std(axis=0, ddof=1) / np.sqrt(n)


def simultaneous_band(D, level=0.95, n_boot=1000, seed=0):
    """Multiplier (Gaussian) bootstrap critical value for a sup-t band.

    Returns (crit, se): band = psi_hat +/- crit * se, valid simultaneously
    over all columns of D. crit >= 1.96 always.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    se = if_se(D)
    Dc = (D - D.mean(axis=0)) / np.clip(se * n, 1e-12, None)  # scale so xi'Dc ~ N(0,1) per col
    sups = np.empty(n_boot)
    for b in range(n_boot):
        xi = rng.standard_normal(n)
        sups[b] = np.max(np.abs(xi @ Dc))
    crit = float(np.quantile(sups, level))
    return max(crit, 1.959964), se
