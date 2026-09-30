"""Cox proportional-hazards versions of the unadjusted and proximal estimators.

The proximal Cox model keeps the first stage of the linear bridge (fitted outcome
proxies What = E[W | A, Z, C], reduced to the directions that vary beyond A) and
replaces the second-stage linear regression by a Cox partial likelihood in
(A, What, C). It is a working model, not an exact bridge for the hazard; the
plug-in standard error ignores first-stage uncertainty, so a bootstrap is
provided for the reported exposures.
"""
import numpy as np
from proximal import _lstsq, proxy_strength


def cox_fit(T, E, X, max_iter=50, tol=1e-9, ridge=1e-8):
    """Breslow partial likelihood by Newton-Raphson. Returns (beta, se)."""
    T = np.asarray(T, float); E = np.asarray(E, float); X = np.atleast_2d(np.asarray(X, float).T).T
    o = np.argsort(-T, kind='stable'); T, E, X = T[o], E[o], X[o]          # descending time: risk set = prefix
    n, p = X.shape; X = X - X.mean(0); beta = np.zeros(p)
    # tie handling (Breslow): every failure at time t shares the risk set {T >= t}
    _, first = np.unique(-T, return_index=True); grp = np.repeat(np.arange(len(first)), np.diff(np.r_[first, n]))
    for _ in range(max_iter):
        eta = X @ beta; w = np.exp(eta - eta.max())
        S0 = np.cumsum(w); S1 = np.cumsum(w[:, None] * X, axis=0); S2 = np.cumsum(w[:, None, None] * X[:, :, None] * X[:, None, :], axis=0)
        last = np.r_[first[1:], n] - 1                                           # last row of each tie group
        S0g, S1g, S2g = S0[last][grp], S1[last][grp], S2[last][grp]
        mu = S1g / S0g[:, None]
        g = (E[:, None] * (X - mu)).sum(0)
        H = (E[:, None, None] * (S2g / S0g[:, None, None] - mu[:, :, None] * mu[:, None, :])).sum(0) + ridge * np.eye(p)
        step = np.linalg.solve(H, g); beta = beta + step
        if np.abs(step).max() < tol: break
    eta = X @ beta; w = np.exp(eta - eta.max()); S0 = np.cumsum(w); S1 = np.cumsum(w[:, None] * X, axis=0)
    S2 = np.cumsum(w[:, None, None] * X[:, :, None] * X[:, None, :], axis=0)
    last = np.r_[first[1:], n] - 1; S0g, S1g, S2g = S0[last][grp], S1[last][grp], S2[last][grp]; mu = S1g / S0g[:, None]
    H = (E[:, None, None] * (S2g / S0g[:, None, None] - mu[:, :, None] * mu[:, None, :])).sum(0) + ridge * np.eye(p)
    return beta, np.sqrt(np.diag(np.linalg.inv(H)))


def cox_effect(T, E, A, C=None):
    X = np.column_stack([A] + ([C] if C is not None and C.shape[1] else []))
    b, se = cox_fit(T, E, X); return b[0], se[0]


def _first_stage(A, W, Z, r, C, min_sv_ratio):
    n = len(A); W = np.atleast_2d(W.T).T; Z = np.atleast_2d(Z.T).T
    ex = [A[:, None]] + ([C] if C is not None and C.shape[1] else [])
    Zf = np.column_stack([np.ones(n)] + ex + [Z]); What = Zf @ _lstsq(Zf, W)
    D0 = np.column_stack([np.ones(n)] + ex)
    _, sv, Vt = np.linalg.svd(What - D0 @ _lstsq(D0, What), full_matrices=False)
    r_eff = max(1, min(r, int((sv >= min_sv_ratio * sv[0]).sum())))
    return What @ Vt[:r_eff].T if r_eff < W.shape[1] else What


def proximal_cox(T, E, A, W, Z, r, C=None, r_nu=None, min_sv_ratio=0.1, n_boot=0, rng=0):
    """Proximal Cox: log hazard ratio per unit of A, se (plug-in, or bootstrap if n_boot > 0), nu."""
    nu = proxy_strength(np.atleast_2d(W.T).T, np.atleast_2d(Z.T).T, A, r if r_nu is None else r_nu, C)
    What = _first_stage(A, W, Z, r, C, min_sv_ratio)
    X = np.column_stack([A, What] + ([C] if C is not None and C.shape[1] else []))
    b, se = cox_fit(T, E, X)
    if n_boot:
        rg = np.random.default_rng(rng); bs = []
        for _ in range(n_boot):
            i = rg.integers(0, len(A), len(A))
            Wb = _first_stage(A[i], W[i], Z[i], r, None if C is None or not C.shape[1] else C[i], min_sv_ratio)
            Xb = np.column_stack([A[i], Wb] + ([C[i]] if C is not None and C.shape[1] else []))
            bs.append(cox_fit(T[i], E[i], Xb)[0][0])
        # Report the bootstrap standard error itself.  Taking max(plugin,
        # bootstrap) silently changes the estimand of the uncertainty summary.
        se = np.std(bs, ddof=1)
        return b[0], se, nu
    return b[0], se[0], nu
