"""DragonNet-style discrete-time survival network + censoring network.

DragonSurv: shared trunk -> propensity head + two outcome heads (hazard logits
over K bins under a=1 and a=0). The final linear layer of each outcome head is
the TDA targeting submodel.

CensNet: separate network for the censoring hazard h_c(k | A, X) over bins
1..K-1 (bin-K censoring is administrative/deterministic and never enters the
IPCW weights for targets t <= K).
"""
import numpy as np
import torch
import torch.nn as nn


def person_bin_arrays(Ttil, Delta, K):
    """at_risk (n,K): 1{Ttil >= k}; dN (n,K): 1{Ttil = k, Delta = 1}."""
    bins = np.arange(1, K + 1)[None, :]
    at_risk = (Ttil[:, None] >= bins).astype(np.float64)
    dN = ((Ttil[:, None] == bins) & (Delta[:, None] == 1)).astype(np.float64)
    return at_risk, dN


class DragonSurv(nn.Module):
    def __init__(self, d_in, K, hidden=64):
        super().__init__()
        self.K = K
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.ELU(),
                                   nn.Linear(hidden, hidden), nn.ELU())
        self.prop_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(),
                                       nn.Linear(hidden, 1))
        self.head1_hidden = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU())
        self.head0_hidden = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU())
        self.head1_out = nn.Linear(hidden, K)  # targeting submodel (arm 1)
        self.head0_out = nn.Linear(hidden, K)  # targeting submodel (arm 0)

    def features(self, X):
        z = self.trunk(X)
        return self.head1_hidden(z), self.head0_hidden(z)

    def forward(self, X):
        z = self.trunk(X)
        phi1 = self.head1_hidden(z)
        phi0 = self.head0_hidden(z)
        return self.head1_out(phi1), self.head0_out(phi0), self.prop_head(z).squeeze(-1)


class PropNet(nn.Module):
    """Standalone propensity network (used when g must be fit separately from
    the outcome trunk, e.g. in nuisance-misspecification scenarios)."""

    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, X):
        return self.net(X).squeeze(-1)


def _select_cols(X, cols):
    """Apply a covariate mask (list of column indices) if one is set."""
    return X if cols is None else X[:, cols]


class CensNet(nn.Module):
    def __init__(self, d_in, K, hidden=64):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(d_in + 1, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, K - 1))

    def forward(self, X, A):
        return self.net(torch.cat([X, A[:, None]], dim=1))


def surv_nll(logits1, logits0, A, at_risk, dN):
    """Discrete-time survival NLL as masked per-bin BCE on the observed arm."""
    logits = A[:, None] * logits1 + (1.0 - A[:, None]) * logits0
    bce = nn.functional.binary_cross_entropy_with_logits(logits, dN, reduction="none")
    return (bce * at_risk).sum(dim=1).mean()


def train_dragonsurv(data, hidden=64, prop_weight=1.0, lr=1e-3, max_epochs=500,
                     patience=25, val_frac=0.2, seed=0, cols=None, verbose=False):
    torch.manual_seed(seed)
    X = torch.as_tensor(_select_cols(data["X"], cols), dtype=torch.float32)
    A = torch.as_tensor(data["A"], dtype=torch.float32)
    K = data["K"]
    at_risk_np, dN_np = person_bin_arrays(data["Ttil"], data["Delta"], K)
    at_risk = torch.as_tensor(at_risk_np, dtype=torch.float32)
    dN = torch.as_tensor(dN_np, dtype=torch.float32)

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    va, tr = perm[:n_val], perm[n_val:]

    net = DragonSurv(X.shape[1], K, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best_val, best_state, bad = np.inf, None, 0
    for epoch in range(max_epochs):
        net.train()
        opt.zero_grad()
        l1, l0, lg = net(X[tr])
        loss = surv_nll(l1, l0, A[tr], at_risk[tr], dN[tr]) + prop_weight * \
            nn.functional.binary_cross_entropy_with_logits(lg, A[tr])
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            l1, l0, lg = net(X[va])
            vloss = (surv_nll(l1, l0, A[va], at_risk[va], dN[va]) + prop_weight *
                     nn.functional.binary_cross_entropy_with_logits(lg, A[va])).item()
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and epoch % 50 == 0:
            print(f"  epoch {epoch}: val {vloss:.4f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    net.cols = cols
    return net


def train_censnet(data, hidden=64, lr=1e-3, max_epochs=500, patience=25,
                  val_frac=0.2, seed=1, cols=None, verbose=False):
    torch.manual_seed(seed)
    X = torch.as_tensor(_select_cols(data["X"], cols), dtype=torch.float32)
    A = torch.as_tensor(data["A"], dtype=torch.float32)
    K = data["K"]
    Ttil, Delta = data["Ttil"], data["Delta"]
    bins = np.arange(1, K)[None, :]  # censoring modeled on bins 1..K-1
    # at risk for censoring at bin k: at risk overall and no event in bin k
    at_risk_c = ((Ttil[:, None] >= bins) &
                 ~((Ttil[:, None] == bins) & (Delta[:, None] == 1))).astype(np.float64)
    dNc = ((Ttil[:, None] == bins) & (Delta[:, None] == 0)).astype(np.float64)
    at_risk_c = torch.as_tensor(at_risk_c, dtype=torch.float32)
    dNc = torch.as_tensor(dNc, dtype=torch.float32)

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    va, tr = perm[:n_val], perm[n_val:]

    net = CensNet(X.shape[1], K, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def nll(idx):
        logits = net(X[idx], A[idx])
        bce = nn.functional.binary_cross_entropy_with_logits(logits, dNc[idx], reduction="none")
        return (bce * at_risk_c[idx]).sum(dim=1).mean()

    best_val, best_state, bad = np.inf, None, 0
    for epoch in range(max_epochs):
        net.train()
        opt.zero_grad()
        loss = nll(tr)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vloss = nll(va).item()
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and epoch % 50 == 0:
            print(f"  cens epoch {epoch}: val {vloss:.4f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    net.cols = cols
    return net


def train_propnet(data, hidden=64, lr=1e-3, max_epochs=500, patience=25,
                  val_frac=0.2, seed=2, cols=None):
    torch.manual_seed(seed)
    X = torch.as_tensor(_select_cols(data["X"], cols), dtype=torch.float32)
    A = torch.as_tensor(data["A"], dtype=torch.float32)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    va, tr = perm[:n_val], perm[n_val:]

    net = PropNet(X.shape[1], hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best_val, best_state, bad = np.inf, None, 0
    for epoch in range(max_epochs):
        net.train()
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(net(X[tr]), A[tr])
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vloss = nn.functional.binary_cross_entropy_with_logits(net(X[va]), A[va]).item()
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    net.cols = cols
    return net


@torch.no_grad()
def nuisances(net, censnet, data, propnet=None, g_bounds=(0.05, 0.95), sc_min=0.05):
    """Fitted nuisance arrays (numpy): h1, h0 (n,K), g (n,), Sc_lag (n,K).

    Sc_lag[:, k-1] = S_c(k-1 | A, X), the probability of remaining uncensored
    through bin k-1 (col 0 is 1 by convention). If propnet is given, g comes
    from it instead of the outcome trunk's propensity head (needed whenever
    the outcome and weight models must be fit on different covariate sets).
    Each network's `cols` covariate mask, if set, is respected.
    """
    X_full = torch.as_tensor(data["X"], dtype=torch.float32)
    A = torch.as_tensor(data["A"], dtype=torch.float32)
    X_out = X_full if getattr(net, "cols", None) is None else X_full[:, net.cols]
    l1, l0, lg = net(X_out)
    h1 = torch.sigmoid(l1).numpy()
    h0 = torch.sigmoid(l0).numpy()
    if propnet is not None:
        X_g = X_full if getattr(propnet, "cols", None) is None else X_full[:, propnet.cols]
        lg = propnet(X_g)
    g = np.clip(torch.sigmoid(lg).numpy(), g_bounds[0], g_bounds[1])
    X_c = X_full if getattr(censnet, "cols", None) is None else X_full[:, censnet.cols]
    hc = torch.sigmoid(censnet(X_c, A)).numpy()  # (n, K-1)
    Sc = np.cumprod(1.0 - hc, axis=1)          # S_c(1)..S_c(K-1)
    Sc_lag = np.hstack([np.ones((X_full.shape[0], 1)), Sc])
    Sc_lag = np.clip(Sc_lag, sc_min, 1.0)
    return h1, h0, g, Sc_lag
