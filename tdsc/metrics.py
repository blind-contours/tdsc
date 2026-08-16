"""Monte Carlo aggregation: bias / variance / MSE / coverage vs truth."""
import numpy as np


def summarize_curve(est, truth, se=None, crit=None):
    """est (R, m) estimates over reps; truth (m,); se (R, m) optional.

    Returns time-averaged metrics; coverage is pointwise unless crit (R,)
    simultaneous critical values are supplied alongside se.
    """
    est = np.asarray(est)
    bias = est.mean(axis=0) - truth
    var = est.var(axis=0, ddof=1)
    mse = ((est - truth) ** 2).mean(axis=0)
    out = dict(bias=float(np.abs(bias).mean()), variance=float(var.mean()),
               mse=float(mse.mean()))
    if se is not None:
        se = np.asarray(se)
        if crit is None:
            cover = (np.abs(est - truth) <= 1.959964 * se).mean(axis=0)
            out["coverage_pointwise"] = float(cover.mean())
            out["ci_width"] = float((2 * 1.959964 * se).mean())
        else:
            crit = np.asarray(crit)[:, None]
            inside = np.abs(est - truth) <= crit * se
            out["coverage_simultaneous"] = float(inside.all(axis=1).mean())
            out["band_width"] = float((2 * crit * se).mean())
    return out


def summarize_scalar(est, truth, se=None):
    est = np.asarray(est)
    out = dict(bias=float(est.mean() - truth), variance=float(est.var(ddof=1)),
               mse=float(((est - truth) ** 2).mean()))
    if se is not None:
        se = np.asarray(se)
        out["coverage"] = float((np.abs(est - truth) <= 1.959964 * se).mean())
        out["ci_width"] = float((2 * 1.959964 * se).mean())
    return out
