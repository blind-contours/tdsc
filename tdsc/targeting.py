"""TDA universal targeting update for the survival-contrast target vector.

Targeting submodel: the final linear layers of the two outcome heads
(head1_out, head0_out), p = 2*K*(hidden+1) parameters.

Per-sample loss gradients wrt these parameters have closed form. For the
discrete-time NLL, for a subject with A_i = a and head features phi_i:
  d l_i / d W_a[k, :] = Y_i(k) (h_a(k|X_i) - dN_i(k)) * phi_i
  d l_i / d b_a[k]    = Y_i(k) (h_a(k|X_i) - dN_i(k))
and zero for the other arm's head.

Each iteration: ridge-project each target's IF onto the gradient span
(one shared Gram solve for all targets), merge directions with the universal
weights w_j = d_j / ||d||_2 (arXiv:2507.12435, Sec 2.3), take a line-searched
step, and recompute nuisance hazards + IFs.

Stopping criterion (TDA-faithful): convergence is assessed on the PROJECTED
EIF means P_n[D_proj,j] -- the estimating equations of the working submodel,
which the restricted parameterization can actually solve -- with tolerance
sd(D_proj,j)/(sqrt(n) log n). The FULL-EIF means P_n[D_j] are tracked as the
diagnostic feeding the one-step residual top-up (they need not vanish inside
a restricted submodel; their projection residual is the gradient-coverage
quantity). The line search minimizes the projected criterion with the
projection coefficients alpha held fixed within the iteration.
"""
import numpy as np
import torch

from .influence import eif_matrix
from .model import person_bin_arrays


def _gradient_matrix(A, at_risk, dN, h1, h0, phi1, phi0):
    """G (n, p): per-sample gradients of the survival NLL wrt targeting params.

    Layout: [W1 (K*F), b1 (K), W0 (K*F), b0 (K)].
    """
    n, K = h1.shape
    F = phi1.shape[1]
    ind1 = (A == 1).astype(np.float64)[:, None]
    R1 = at_risk * (h1 - dN) * ind1
    R0 = at_risk * (h0 - dN) * (1.0 - ind1)
    G1w = np.einsum("nk,nf->nkf", R1, phi1).reshape(n, K * F)
    G0w = np.einsum("nk,nf->nkf", R0, phi0).reshape(n, K * F)
    return np.hstack([G1w, R1, G0w, R0])


def _apply_step(net, direction, gamma, K, F):
    """theta_targ <- theta_targ - gamma * direction (mapped into head tensors)."""
    i = 0
    with torch.no_grad():
        for out_layer in (net.head1_out, net.head0_out):
            dW = direction[i:i + K * F].reshape(K, F)
            i += K * F
            db = direction[i:i + K]
            i += K
            out_layer.weight -= gamma * torch.as_tensor(dW, dtype=torch.float32)
            out_layer.bias -= gamma * torch.as_tensor(db, dtype=torch.float32)


@torch.no_grad()
def _hazards(net, X_t):
    l1, l0, _ = net(X_t)
    return torch.sigmoid(l1).numpy(), torch.sigmoid(l0).numpy()


def tda_target(net, data, g, Sc_lag, ridge=1e-2, max_iter=50, gamma0=1.0,
               max_halvings=12, verbose=False):
    """Run the universal TDA targeting loop. Mutates `net` (final layers).

    g and Sc_lag are held fixed (only the outcome heads are targeted).
    Returns dict with final hazards, EIF matrix, and iteration diagnostics.
    """
    X_t = torch.as_tensor(data["X"], dtype=torch.float32)
    if getattr(net, "cols", None) is not None:
        X_t = X_t[:, net.cols]
    A, K = data["A"], data["K"]
    n = X_t.shape[0]
    at_risk, dN = person_bin_arrays(data["Ttil"], data["Delta"], K)
    with torch.no_grad():
        phi1, phi0 = net.features(X_t)
    phi1, phi0 = phi1.numpy().astype(np.float64), phi0.numpy().astype(np.float64)
    F = phi1.shape[1]

    history = []
    converged = False
    alpha = None
    D_proj = None
    h1, h0 = _hazards(net, X_t)
    D = eif_matrix(A, at_risk, dN, h1, h0, g, Sc_lag)
    for it in range(max_iter):
        G = _gradient_matrix(A, at_risk, dN, h1, h0, phi1, phi0)
        Gram = G.T @ G
        lam = ridge * np.mean(np.diag(Gram)) + 1e-10
        Gram[np.diag_indices_from(Gram)] += lam
        alpha = np.linalg.solve(Gram, G.T @ D)          # (p, 2K), all targets at once
        D_proj = G @ alpha                              # working-submodel EIFs (n, 2K)
        d_proj = D_proj.mean(axis=0)
        tol_proj = D_proj.std(axis=0, ddof=1) / (np.sqrt(n) * np.log(n))
        crit = float(np.sum(d_proj ** 2))
        history.append(dict(
            iter=it, crit=crit,
            max_abs_pnd_proj=float(np.max(np.abs(d_proj))),
            max_abs_pnd_full=float(np.max(np.abs(D.mean(axis=0)))),
            proj_resid_rel=float(np.linalg.norm(D - D_proj) / np.linalg.norm(D))))
        if verbose:
            print(f"  TDA it {it}: max|PnDproj| {np.max(np.abs(d_proj)):.5f}, "
                  f"crit {crit:.3e}")
        if np.all(np.abs(d_proj) <= tol_proj):
            converged = True
            break
        norm = np.linalg.norm(d_proj)
        if norm < 1e-12:
            break
        w = d_proj / norm
        direction = alpha @ w                           # universal direction (p,)

        # backtracking line search on the projected criterion, alpha held fixed
        gamma, accepted = gamma0, False
        for _ in range(max_halvings):
            _apply_step(net, direction, gamma, K, F)
            h1_new, h0_new = _hazards(net, X_t)
            G_new = _gradient_matrix(A, at_risk, dN, h1_new, h0_new, phi1, phi0)
            crit_new = float(np.sum((G_new @ alpha).mean(axis=0) ** 2))
            if crit_new < crit:
                h1, h0 = h1_new, h0_new
                D = eif_matrix(A, at_risk, dN, h1, h0, g, Sc_lag)
                accepted = True
                break
            _apply_step(net, direction, -gamma, K, F)   # undo, halve
            gamma *= 0.5
        if not accepted:
            break
    # refresh the projection at the final fit so D_proj/alpha match (h1, h0)
    G = _gradient_matrix(A, at_risk, dN, h1, h0, phi1, phi0)
    Gram = G.T @ G
    lam = ridge * np.mean(np.diag(Gram)) + 1e-10
    Gram[np.diag_indices_from(Gram)] += lam
    alpha = np.linalg.solve(Gram, G.T @ D)
    D_proj = G @ alpha
    d_proj = D_proj.mean(axis=0)
    tol_proj = D_proj.std(axis=0, ddof=1) / (np.sqrt(n) * np.log(n))
    converged = converged or bool(np.all(np.abs(d_proj) <= tol_proj))
    d_full = D.mean(axis=0)
    tol_full = D.std(axis=0, ddof=1) / (np.sqrt(n) * np.log(n))
    return dict(h1=h1, h0=h0, D=D, D_proj=D_proj, alpha=alpha, history=history,
                converged=converged, final_pnd_proj=d_proj, final_tol_proj=tol_proj,
                final_pnd=d_full, final_tol=tol_full)
