"""Proximal causal inference with proxies learned from an idopNetwork.

Pipeline
    levels X (patients ordered by the niche index)
      -> population power curves C(s)                      [idop_core]
      -> patient deviations U = X - C(s)
      -> latent state F, loadings Lambda, rank r  (PCA of U)
      -> support graph from patient-level edge selection    [idop_core.edge_select]
      -> proxy roles Z (adjacent to A) and W (disconnected from A and Z)
      -> proximal linear bridge; proxy strength nu; sensitivity threshold

The latent state is taken from the deviations rather than from the curve-based
reconstruction: with X = C(s) + U and U = U Theta + F Lambda' + E,
U = F [Lambda'(I - Theta)^{-1}] + E (I - Theta)^{-1}, so the column space of U's
low-rank part is F whatever Theta is.
"""
from collections import deque
import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- latent state
def latent_state(U, rmax=8):
    """Rank by the largest multiplicative gap of the spectrum; F, Lambda by SVD.

    Returns dict(r, F, Lam, sv, rho) where rho[i] = ||Xi_i|| / ||U_i|| is the
    residual norm fraction of variable i after its rank-r projection;
    rho[i]**2 is the corresponding residual variance fraction.
    """
    Uc = U - U.mean(0)
    Uu, sv, Vt = np.linalg.svd(Uc, full_matrices=False)
    svp = sv[sv > 1e-10]
    if svp.size <= 1:
        r = max(int(svp.size), 1)
    else:
        k = min(rmax, svp.size - 1)
        r = int(np.argmax(svp[:k] / np.maximum(svp[1:k + 1], 1e-12)) + 1)
    F = Uu[:, :r] * sv[:r]
    Lam = Vt[:r].T
    Xi = Uc - F @ np.linalg.lstsq(F, Uc, rcond=None)[0]
    rho = np.linalg.norm(Xi, axis=0) / np.maximum(np.linalg.norm(Uc, axis=0), 1e-12)
    return dict(r=r, F=F, Lam=Lam, sv=sv, rho=rho, Xi=Xi)


# --------------------------------------------------------------------------- support graph
def adjacency_from_supports(edge_supports, names):
    """edge_select output (target, source='{a,b}') -> boolean matrix A[source, target]."""
    p = len(names); A = np.zeros((p, p), bool); idx = {n: i for i, n in enumerate(names)}
    for row in edge_supports.itertuples():
        srcs = row.source.strip('{}')
        for s in ([] if not srcs else srcs.split(',')):
            A[idx[s.strip()], idx[row.target]] = True
    return A


def components(adj):
    p = len(adj); comp = -np.ones(p, int); c = 0
    for s in range(p):
        if comp[s] >= 0: continue
        q = deque([s]); comp[s] = c
        while q:
            u = q.popleft()
            for v in np.flatnonzero(adj[u]):
                if comp[v] < 0: comp[v] = c; q.append(v)
        c += 1
    return comp


def _descendants(directed, v, block):
    """Nodes reachable from v along directed edges without passing through `block`."""
    seen = set(); q = deque([v])
    while q:
        u = q.popleft()
        for w in np.flatnonzero(directed[u]):
            if w == block or w in seen or w == v: continue
            seen.add(w); q.append(w)
    return seen


def proxy_roles(undirected, a0, pi_absent, lam_norm, r, lam_tol=1e-8, directed=None, limit_to_treatment=True):
    """Nominate Z adjacent to the exposure, optionally filtered by ``pi_absent``.
    When a genuinely causal directed support is supplied, descendants can also be filtered;
    the primary nodewise-support pipeline does not supply such a direction.
    W: no path to the exposure or to any Z, nonzero latent loading.
    ``limit_to_treatment=True`` retains the earlier balanced-pool rule, selecting
    low-degree W candidates when |W| > |Z|. The revised primary estimator passes
    False and retains the complete eligible separated pool.
    """
    p = undirected.shape[0]
    def z_ok(v):
        if not pi_absent[v]: return False
        if directed is None: return True
        return all(pi_absent[u] for u in _descendants(directed, v, a0))
    Z = [v for v in range(p) if v != a0 and undirected[a0, v] and z_ok(v)]
    if not Z: return [], []
    comp = components(undirected)
    bad = {comp[a0]} | {comp[z] for z in Z}
    W = [v for v in range(p) if v != a0 and v not in Z and comp[v] not in bad and lam_norm[v] > lam_tol]
    if limit_to_treatment and len(W) > len(Z):
        W = sorted(W, key=lambda w: undirected[w].sum())[:len(Z)]
    return Z, sorted(W)


def screen(Z, W, r, overcomplete=False):
    return len(W) >= r and len(Z) >= (r if overcomplete else len(W))


# --------------------------------------------------------------------------- outcome equation
def outcome_equation(Y, X, F, Cov=None, ridge=1.0, alpha=0.05, equivalence_margin=0.15,
                     return_diagnostics=False):
    """Screen approximately null outcome coefficients by equivalence testing.

    ``equivalence_margin`` is expressed in outcome-SD units per one SD of a
    candidate variable.  A coefficient is called approximately absent only
    when its two one-sided-test confidence interval lies wholly inside
    ``[-equivalence_margin, equivalence_margin]``.  This deliberately differs
    from treating a non-significant test against zero as evidence of absence.

    The ridge covariance uses the linear-smoother sandwich
    ``G^-1 X'X G^-1``.  Set ``return_diagnostics`` to return the standardized
    coefficients and standard errors with the Boolean decisions.
    """
    if equivalence_margin <= 0:
        raise ValueError('equivalence_margin must be positive')
    if not 0 < alpha < 0.5:
        raise ValueError('alpha must lie between 0 and 0.5 for TOST')
    n = len(Y); parts = [np.ones(n)]
    if Cov is not None and Cov.shape[1]: parts.append(Cov)
    parts += [F, X - X.mean(0)]
    Dm = np.column_stack(parts)
    G = Dm.T @ Dm + ridge * np.eye(Dm.shape[1])
    b = np.linalg.solve(G, Dm.T @ Y)
    k0 = Dm.shape[1] - X.shape[1]
    resid = Y - Dm @ b; s2 = (resid @ resid) / max(n - Dm.shape[1], 1)
    Gi = np.linalg.inv(G)
    V = s2 * Gi @ (Dm.T @ Dm) @ Gi
    se = np.sqrt(np.maximum(np.diag(V)[k0:], 0))
    x_sd = np.std(X, axis=0, ddof=0)
    y_sd = max(float(np.std(Y, ddof=0)), 1e-12)
    effect_std = b[k0:] * x_sd / y_sd
    se_std = se * x_sd / y_sd
    # A level-alpha TOST corresponds to a (1 - 2 alpha) equivalence CI.
    z = stats.norm.ppf(1 - alpha)
    absent = np.abs(effect_std) + z * se_std < equivalence_margin
    if return_diagnostics:
        return dict(absent=absent, effect_std=effect_std, se_std=se_std,
                    margin=equivalence_margin, alpha=alpha)
    return absent


# --------------------------------------------------------------------------- estimators
def bridge_information_rank(W, Z, A, C=None, alpha=None, rmax=8):
    """Estimate the rank of Cov(W, Z | 1,A,C) by sequential canonical roots.

    The vanishing default threshold alpha_n=min(0.01,1/n) separates O_p(1)
    null-root statistics from O(n) identifying signals. Bartlett's calibration
    supplies a finite-sample reference under Gaussian proxy errors; consistency
    uses a root-n covariance expansion and separated population roots.
    This is an outcome-blind rank decision, separate from protein PCA rank.
    """
    n = len(W); A = None if A is None else np.asarray(A, float)
    W = np.asarray(W, float).reshape(n, -1); Z = np.asarray(Z, float).reshape(n, -1)
    D = np.column_stack([np.ones(n)] + ([] if A is None else [A]) + ([C] if C is not None and C.shape[1] else []))
    rw = W - D @ _lstsq(D, W); rz = Z - D @ _lstsq(D, Z)
    def whitening(X):
        values, vectors = np.linalg.eigh(X.T @ X / n)
        keep = values > max(float(values.max()) * 1e-10, 1e-12)
        return vectors[:, keep] / np.sqrt(values[keep])
    hw, hz = whitening(rw), whitening(rz)
    if not hw.shape[1] or not hz.shape[1]:
        return dict(r=0, roots=np.array([]), pvalues=np.array([]), alpha=alpha or min(.01, 1/n))
    roots = np.clip(np.linalg.svd(hw.T @ (rw.T @ rz / n) @ hz, compute_uv=False), 0, 1-1e-12)
    pw, pz = hw.shape[1], hz.shape[1]
    scale = max(n - np.linalg.matrix_rank(D) - (pw+pz+1)/2, 1)
    level = min(.01, 1/n) if alpha is None else float(alpha)
    if not 0 < level < 1: raise ValueError('rank alpha must lie in (0,1)')
    pvalues = np.array([stats.chi2.sf(-scale*np.log1p(-roots[k:]**2).sum(),
                                    (pw-k)*(pz-k)) for k in range(len(roots))])
    rank = 0
    for k in range(min(len(roots), rmax)):
        if pvalues[k] >= level: break
        rank += 1
    return dict(r=rank, roots=roots, pvalues=pvalues, alpha=level)


def readout_information_rank(X, undirected, a0, C=None, eligible=None):
    """Joint rank Cov(W,[A,Z]|C), before exposure residualisation.

    Under P2, W's errors are orthogonal to both A and Z. This joint rank
    includes a systemic direction primarily measured by A; conditioning on A
    can make that direction weak in Z. Matching joint and residual ranks
    checks whether Z supplies the information needed to disentangle it.
    """
    eligible=np.ones(X.shape[1],bool) if eligible is None else np.asarray(eligible,bool)
    Z,W=proxy_roles(undirected,a0,np.ones(X.shape[1],bool),eligible.astype(float),1,limit_to_treatment=False)
    if not Z or not W:return dict(r=0,views=[W,[a0]+Z],roots=np.array([]),pvalues=np.array([]))
    info=bridge_information_rank(X[:,W],X[:,[a0]+Z],None,C)
    info['views']=[W,[a0]+Z]
    return info


def _lstsq(D, y):
    return np.linalg.lstsq(D, y, rcond=None)[0]


def ols_effect(Y, A, C=None):
    D = np.column_stack([np.ones(len(Y)), A] + ([C] if C is not None and C.shape[1] else []))
    b = _lstsq(D, Y); res = Y - D @ b
    V = (res @ res) / (len(Y) - D.shape[1]) * np.linalg.pinv(D.T @ D)
    return b[1], np.sqrt(V[1, 1])


def proxy_strength(W, Z, A, r, C=None):
    """nu = sigma_r(Cov(W, Z | A, C)) / (||sd W|| ||sd Z||): the r-th singular value of
    the first-stage cross-covariance on a correlation scale (proximal analogue of
    the Cragg-Donald statistic)."""
    n = len(A); W = np.atleast_2d(W.T).T; Z = np.atleast_2d(Z.T).T
    D = np.column_stack([np.ones(n), A] + ([C] if C is not None and C.shape[1] else []))
    rW = W - D @ _lstsq(D, W); rZ = Z - D @ _lstsq(D, Z)
    sv = np.linalg.svd(rW.T @ rZ / n, compute_uv=False)
    k = min(r, len(sv)); den = np.linalg.norm(rW.std(0)) * np.linalg.norm(rZ.std(0))
    return float(sv[k - 1] / den) if den > 1e-12 else 0.0


def proximal_bridge(Y, A, W, Z, r, C=None, r_nu=None, min_sv_ratio=0.1):
    """Linear outcome bridge: regress W on (A, Z, C), truncate the fitted W to its
    leading r directions (E[W|A,Z] has rank r in the population), regress Y on
    (A, What, C). Returns (effect, se, nu); nu is evaluated at rank r_nu (default r),
    so a spare truncation direction does not change the reported proxy strength."""
    n = len(Y); W = np.atleast_2d(W.T).T; Z = np.atleast_2d(Z.T).T
    nu = proxy_strength(W, Z, A, r if r_nu is None else r_nu, C)      # on the proxies as selected
    ex = [A[:, None]] + ([C] if C is not None and C.shape[1] else [])
    Zf = np.column_stack([np.ones(n)] + ex + [Z])
    What = Zf @ _lstsq(Zf, W)
    # keep only the directions of E[W|A,Z] that vary beyond (1, A, C): a direction with a
    # negligible singular value there makes the second stage collinear with A
    D0 = np.column_stack([np.ones(n)] + ex)
    _, sv, Vt = np.linalg.svd(What - D0 @ _lstsq(D0, What), full_matrices=False)
    r_eff = max(1, min(r, int((sv >= min_sv_ratio * sv[0]).sum())))
    if r_eff < W.shape[1]:
        P = Vt[:r_eff].T
        W, What = W @ P, What @ P
    Xen = np.column_stack([np.ones(n)] + ex + [W]); Xht = np.column_stack([np.ones(n)] + ex + [What])
    b = _lstsq(Xht, Y); res = Y - Xen @ b
    s2 = (res @ res) / max(n - Xht.shape[1], 1)
    V = s2 * np.linalg.pinv(Xht.T @ Xht)
    return b[1], np.sqrt(max(V[1, 1], 0)), nu


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.minimum(q, 1)
