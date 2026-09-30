"""Data generating process: a latent-factor regulatory system with designated proxy blocks.

Static (cross-sectional) reading of the latent-factor model:

    X = X @ Theta + F @ Lambda.T + E       (psi = identity, S = 1)

Patients are ordered by a niche index s; F is the unobserved systemic state U.
Variable layout (p total):

    [0 .. nW-1]        W-block: readouts of F only, no observed parents
    [nW .. nW+nZ-1]    Z-block: children of A, also load on F
    [a0 = nW+nZ]       A: the exposure, driven by F alone so that U = F is the
                       complete unmeasured confounder
    [a0+1 .. p-1]      O-block: background regulatory DAG, some with direct
                       effects on Y

delta sets the magnitude of the violations of the proximal conditions:
    (P2): a delta-sized edge A -> W_j and Z_0 -> W_j
    (P1): a delta-sized direct coefficient of Z_0 on Y

The structural parameters are fixed once by make_system; draw() then samples
patients from that fixed system, so that replication variability is sampling
variability alone.
"""

import numpy as np

N_W, N_Z, N_O, R = 6, 6, 18, 2


def make_system(delta=0.0, gamma=0.5, n_w=N_W, n_z=N_Z, n_o=N_O, r=R,
                lam_conf=1.0, lat_o=1.0, seed=0, delta_p1=None, delta_p2=None,
                n_viol=None):
    """delta_p1 sets the (P1) violation (Z_0 -> Y directly); delta_p2 the
    (P2) violations (A -> W, Z_0 -> W). Each defaults to delta."""
    delta_p1 = delta if delta_p1 is None else delta_p1
    delta_p2 = delta if delta_p2 is None else delta_p2
    # n_viol: how many of the variables disconnected from A carry the (P2)
    # violation. Default is the W-block alone (6 of 24); larger values spill
    # the violation onto the O-block, raising the invalid fraction among the
    # candidates a selection rule has to choose from.
    rng = np.random.default_rng(seed)
    p = n_w + n_z + 1 + n_o
    iW = np.arange(n_w)
    iZ = np.arange(n_w, n_w + n_z)
    a0 = n_w + n_z
    iO = np.arange(a0 + 1, p)

    Theta = np.zeros((p, p))      # Theta[i, j] = directed effect of i on j
    Lam = np.zeros((p, r))

    # background regulatory DAG over the O-block
    # lat_o scales how much of the background network's variation is latent.
    # It controls the identifiability of the sparse-plus-low-rank split: the
    # predictors' idiosyncratic share is the only thing separating X @ Theta
    # from the low-rank term, since X is itself dominated by F.
    Lam[iO] = lat_o * rng.uniform(0.5, 1.1, (n_o, r)) * rng.choice([-1, 1], (n_o, r))
    for pos, j in enumerate(iO):
        for k in iO[pos + 1:pos + 3]:
            if rng.random() < 0.6:
                Theta[j, k] = rng.uniform(0.3, 0.7) * rng.choice([-1, 1])

    # exposure: latent-driven only. Loadings kept positive, and lam_Y below
    # likewise, so the confounding does not cancel across latent directions
    # and the naive association is systematically wrong.
    Lam[a0] = rng.uniform(0.9, 1.3, r)

    # Z-block: strong children of A, strong loadings -> proxy relevance
    Lam[iZ] = rng.uniform(0.8, 1.3, (n_z, r)) * rng.choice([-1, 1], (n_z, r))
    Theta[a0, iZ] = rng.uniform(0.8, 1.2, n_z) * rng.choice([-1, 1], n_z)

    # W-block: readouts of the latent state
    Lam[iW] = rng.uniform(0.9, 1.4, (n_w, r)) * rng.choice([-1, 1], (n_w, r))

    # direct outcome effects on part of the O-block; confounding with a fixed
    # sign so that the naive association is systematically biased
    pi = np.zeros(p)
    pi[iO[3:6]] = rng.uniform(0.5, 0.9, 3)
    lam_Y = lam_conf * rng.uniform(0.8, 1.2, r)

    # ---- the delta-sized violations, written last so that every other
    # structural parameter is identical across the delta sweep
    viol = np.r_[iW, iO][: (n_w if n_viol is None else n_viol)]
    Theta[a0, viol] = delta_p2
    Theta[iZ[0], viol] = delta_p2
    pi[iZ[0]] = delta_p1

    return dict(Theta=Theta, Lam=Lam, pi=pi, lam_Y=lam_Y, gamma=gamma,
                delta=delta, r=r, p=p, a0=a0, iW=iW, iZ=iZ, iO=iO,
                n_w=n_w, n_z=n_z, n_o=n_o)


def draw(sys, n, rng=None, noise=1.0, noise_proxy=0.5, lat_x=1.0,
         smooth_latent=False):
    rng = np.random.default_rng(rng)
    p, r, a0 = sys['p'], sys['r'], sys['a0']
    iW, iZ, iO = sys['iW'], sys['iZ'], sys['iO']
    Theta, Lam = sys['Theta'], sys['Lam']

    # Main simulations use iid Gaussian patients, matching Theorem 1.  A
    # smooth trajectory remains available as a dependence stress test; it
    # requires a block rather than iid bootstrap.
    s = np.sort(rng.uniform(0, 1, n))
    if smooth_latent:
        bw = max(3, n // 40)
        ker = np.exp(-0.5 * (np.arange(-3 * bw, 3 * bw + 1) / bw) ** 2)
        ker /= np.sqrt((ker ** 2).sum())
        F = np.empty((n, r))
        for q in range(r):
            F[:, q] = np.convolve(rng.standard_normal(n + 6 * bw), ker, mode='valid')[:n]
    else:
        F = rng.standard_normal((n, r))
    F -= F.mean(0)
    F /= F.std(0)

    X = np.zeros((n, p))
    # A and Z are generated first so that O-block variables carrying a (P2)
    # violation (n_viol > n_w) can receive their edges from A and Z_0.
    X[:, a0] = lat_x * (F @ Lam[a0]) + noise * rng.standard_normal(n)
    X[:, iZ] = (np.outer(X[:, a0], Theta[a0, iZ]) + lat_x * (F @ Lam[iZ].T)
                + noise_proxy * rng.standard_normal((n, len(iZ))))

    for pos, j in enumerate(iO):
        anc = iO[:pos]
        val = X[:, anc] @ Theta[anc, j] if pos else 0.0
        val = val + X[:, a0] * Theta[a0, j] + X[:, iZ[0]] * Theta[iZ[0], j]
        X[:, j] = val + lat_x * (F @ Lam[j]) + noise * rng.standard_normal(n)

    X[:, iW] = (lat_x * (F @ Lam[iW].T)
                + np.outer(X[:, a0], Theta[a0, iW])
                + np.outer(X[:, iZ[0]], Theta[iZ[0], iW])
                + noise_proxy * rng.standard_normal((n, len(iW))))

    Y = (sys['gamma'] * X[:, a0] + X @ sys['pi'] + F @ sys['lam_Y']
         + noise * rng.standard_normal(n))

    # Derivative response (not used by the level-based pipeline; kept for completeness): each node's rate of
    # change responds to the levels of its parents plus the latent state. The
    # response D and the design X are distinct matrices, which is what makes the
    # sparse-plus-low-rank separation identifiable: with D = X the degenerate
    # solution Theta = 0, M = low-rank part of X is available and cheap, because
    # X is itself dominated by the latent factor.
    D = X @ Theta + F @ Lam.T + noise * rng.standard_normal((n, p))
    return dict(X=X, Y=Y, F=F, s=s, D=D)
