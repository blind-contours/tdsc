"""Data-generating process for the TDSC simulations.

Discrete-time survival on bins k = 1..K with:
  - confounded binary treatment A ~ Bernoulli(g0(X)),
  - heterogeneous treatment effect tau(X) on the log-hazard scale that changes
    sign across the population (qualitative interaction),
  - informative right censoring: the censoring hazard depends on both A and X.

Within-bin ordering: the event opportunity comes first, then censoring, so an
event in bin k is always observed for a subject at risk at k. Subjects who
survive all K bins are administratively censored at K.

Observed data per subject: O = (X, A, Ttil, Delta) with Ttil in {1..K} and
Delta = 1 for an event, 0 for censoring (including administrative).
"""
import numpy as np

D_X = 10


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def propensity(X):
    return np.clip(_expit(0.9 * X[:, 0] - 0.7 * X[:, 1] + 0.6 * X[:, 2]), 0.05, 0.95)


def tau(X):
    """Log-hazard treatment effect; negative = protective. Sign-varying."""
    return -1.0 + 0.8 * X[:, 0] + 0.6 * X[:, 2]


def event_hazard(k, A, X, K):
    base = -3.1 + 2.0 * k / K
    lin = 0.7 * X[:, 0] + 0.5 * X[:, 1] + 0.4 * X[:, 2] - 0.3 * X[:, 3]
    return np.clip(_expit(base + lin + A * tau(X)), 1e-4, 1.0 - 1e-4)


def cens_hazard(k, A, X, K):
    return np.clip(_expit(-3.6 + 0.5 * X[:, 0] + 0.4 * X[:, 1] + 0.4 * A + 0.3 * X[:, 4]),
                   1e-4, 1.0 - 1e-4)


def simulate(n, K=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D_X))
    g = propensity(X)
    A = rng.binomial(1, g)

    Ttil = np.full(n, K, dtype=int)
    Delta = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    for k in range(1, K + 1):
        h = event_hazard(k, A, X, K)
        ev = alive & (rng.random(n) < h)
        Ttil[ev] = k
        Delta[ev] = 1
        alive &= ~ev
        if k < K:
            hc = cens_hazard(k, A, X, K)
            ce = alive & (rng.random(n) < hc)
            Ttil[ce] = k
            alive &= ~ce
    return dict(X=X, A=A, Ttil=Ttil, Delta=Delta, K=K)


def true_survival(K=30, n_mc=400_000, seed=12345):
    """S_a(t_k) = E_X prod_{j<=k} (1 - h(j|a,X)), exact given X draws."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_mc, D_X))
    out = np.zeros((2, K))
    for a in (0, 1):
        Aa = np.full(n_mc, a)
        S = np.ones(n_mc)
        for k in range(1, K + 1):
            S = S * (1.0 - event_hazard(k, Aa, X, K))
            out[a, k - 1] = S.mean()
    return out  # out[a, k-1] = S_a(t_k)


def true_estimands(K=30, n_mc=400_000, seed=12345):
    S = true_survival(K=K, n_mc=n_mc, seed=seed)
    benefit = S[1] - S[0]
    rmst_diff = benefit.sum()  # bin width = 1
    return dict(S1=S[1], S0=S[0], benefit=benefit, rmst_diff=rmst_diff)
