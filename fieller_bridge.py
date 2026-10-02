"""Concentrated bridge moments and Fieller confidence sets for one weak direction.

Joint information identifies a regular nuisance readout block. Concentrate
that block to obtain N - tau D = 0. Bootstrap (N,D), never N/D, and invert the
studentised moment inequality. Proxy roles/rank and finite instrument pivots
are fixed for this construction-conditioned inference; P and censoring refit.
"""
import numpy as np
from scipy.linalg import qr
from scipy.stats import norm


def projection(A, W, Z, r, C=None):
    n = len(A)
    base = np.column_stack([np.ones(n)]+([C] if C is not None and C.shape[1] else []))
    M = np.column_stack([base, A, Z])
    fitted = M@np.linalg.lstsq(M, W, rcond=None)[0]
    centered = fitted-base@np.linalg.lstsq(base, fitted, rcond=None)[0]
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    if r < 1 or r > len(sv) or sv[r-1] <= max(sv[0]*1e-8, 1e-12):
        raise ValueError('Joint readout direction unavailable')
    return W@vt[:r].T


def views(A, W, Z, r, C=None):
    n = len(A)
    base = np.column_stack([np.ones(n)]+([C] if C is not None and C.shape[1] else []))
    R = projection(A, W, Z, r, C)
    H = np.column_stack([A, R])
    Q = np.column_stack([A, Z])
    return base, H-base@np.linalg.lstsq(base, H, rcond=None)[0], Q-base@np.linalg.lstsq(base, Q, rcond=None)[0]


def choose_pivots(A, W, Z, r, C=None):
    base, H, Q = views(A, W, Z, r, C)
    R = H[:, 1:]
    if Q.shape[1] <= r:
        raise ValueError('Need r treatment coordinates for one exposure contrast')
    # Select nuisance coordinates only through the strong joint readout block.
    # Every remaining moment is retained; weak conditional moments are not ranked.
    _, _, rows = qr((Q.T@R).T, pivoting=True, mode='economic')
    nuisance = rows[:r]
    contrasts = sorted(set(range(Q.shape[1]))-set(nuisance))
    B = (Q[:, nuisance].T@R)/len(A)
    sv = np.linalg.svd(B, compute_uv=False)
    if sv[-1] <= max(sv[0]*1e-8, 1e-12):
        raise ValueError('Regular nuisance block unavailable')
    return dict(nuisance=nuisance.tolist(), contrasts=contrasts,
                nuisance_sv=float(sv[-1]), nuisance_condition=float(sv[0]/sv[-1]))


def moments(Y, A, W, Z, r, C, pivots):
    base, H, Q = views(A, W, Z, r, C)
    y = Y-base@np.linalg.lstsq(base, Y, rcond=None)[0]
    a, R = H[:, 0], H[:, 1:]
    T = Q[:, pivots['nuisance']]
    U = Q[:, pivots['contrasts']]
    B = T.T@R/len(A)
    sv = np.linalg.svd(B, compute_uv=False)
    if sv[-1] <= max(sv[0]*1e-8, 1e-12):
        raise ValueError('Nuisance block numerically unavailable')
    L = np.linalg.solve(B.T, R.T@U/len(A)).T
    N = U.T@y/len(A)-L@(T.T@y/len(A))
    D = U.T@a/len(A)-L@(T.T@a/len(A))
    return np.concatenate([N, D])


def scalar_confidence_set(nd, covariance, alpha=.05):
    """Solve a t^2 + b t + c <= 0, retaining unbounded/disconnected sets."""
    N, D = np.asarray(nd, float)
    V = np.asarray(covariance, float)
    z2 = norm.ppf(1-alpha/2)**2
    a = D*D-z2*V[1, 1]
    b = -2*N*D+2*z2*V[0, 1]
    c = N*N-z2*V[0, 0]
    scale = max(abs(a), abs(b), abs(c), 1e-300)
    tol = scale*1e-12
    if abs(a) <= tol:
        if abs(b) <= tol:
            intervals = [(-np.inf, np.inf)] if c <= tol else []
        else:
            cut = -c/b
            intervals = [(-np.inf, cut)] if b > 0 else [(cut, np.inf)]
    else:
        discriminant = b*b-4*a*c
        if discriminant < -1e-12*max(b*b, abs(4*a*c), 1e-300):
            intervals = [] if a > 0 else [(-np.inf, np.inf)]
        else:
            root = np.sqrt(max(discriminant, 0.))
            # Stable roots avoid cancellation when N and D differ in scale.
            q = -.5*(b+np.copysign(root, b))
            roots = sorted([q/a, c/q]) if abs(q) > tol else sorted([(-b-root)/(2*a), (-b+root)/(2*a)])
            intervals = [(roots[0], roots[1])] if a > 0 else [(-np.inf, roots[0]), (roots[1], np.inf)]
    kind = ('empty' if not intervals else 'bounded' if len(intervals) == 1 and np.isfinite(intervals).all()
            else 'all_real' if intervals == [(-np.inf, np.inf)] else 'disconnected' if len(intervals) == 2 else 'half_line')
    return dict(kind=kind, intervals=intervals, N=float(N), D=float(D),
                quadratic=[float(a), float(b), float(c)])


def intersect(left, right):
    pieces = sorted((max(a, c), min(b, d)) for a, b in left for c, d in right
                    if max(a, c) <= min(b, d))
    merged = []
    for lo, hi in pieces:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def confidence_set(nd, covariance, alpha=.05):
    nd, V = np.asarray(nd, float), np.asarray(covariance, float)
    m = len(nd)//2
    if m < 1 or len(nd) != 2*m or V.shape != (2*m, 2*m):
        raise ValueError('Need paired vectors N and D and their joint covariance')
    N, D = nd[:m], nd[m:]
    intervals, components = [(-np.inf, np.inf)], []
    for j in range(m):
        ix = [j, m+j]
        component = scalar_confidence_set(nd[ix], V[np.ix_(ix, ix)], alpha/m)
        components.append(component)
        intervals = intersect(intervals, component['intervals'])
    kind = ('empty' if not intervals else 'bounded' if len(intervals) == 1 and np.isfinite(intervals).all()
            else 'all_real' if intervals == [(-np.inf, np.inf)] else 'disconnected' if len(intervals) > 1 else 'half_line')
    result = dict(kind=kind, intervals=intervals, N=N.tolist(), D=D.tolist(),
                  components=components, contrasts=m, alpha=float(alpha))
    if m == 1:
        result['quadratic'] = components[0]['quadratic']
    return result


def point_and_gradient(nd):
    nd = np.asarray(nd, float)
    m = len(nd)//2
    N, D = nd[:m], nd[m:]
    s = float(D@D)
    if s <= 1e-24:
        return np.nan, np.full(2*m, np.nan)
    tau = float(D@N/s)
    return tau, np.concatenate([D/s, N/s-2*tau*D/s])


def contains(result, value):
    return any(lo <= value <= hi for lo, hi in result['intervals'])


def fit(Y, A, W, Z, r, C=None, draws=None, responses=None, alpha=.05):
    C = np.zeros((len(A), 0)) if C is None else C
    pivots = choose_pivots(A, W, Z, r, C)
    nd = moments(Y, A, W, Z, r, C, pivots)
    values = []
    for b, ix in enumerate(draws or []):
        try:
            yy = Y[ix] if responses is None else responses[b]
            value = moments(yy, A[ix], W[ix], Z[ix], r, C[ix], pivots)
            if np.isfinite(value).all():
                values.append(value)
        except (ValueError, np.linalg.LinAlgError):
            pass
    if len(values) < max(20, .975*len(draws or [])):
        raise ValueError('Nuisance-moment bootstrap incomplete')
    covariance = np.cov(np.array(values).T, ddof=1)
    result = confidence_set(nd, covariance, alpha)
    result.update(pivots=pivots, covariance=covariance.tolist(), boot_success=len(values),
                  estimate=point_and_gradient(nd)[0],
                  inference='fixed selected molecular roles/rank and finite instrument pivots; P and censoring refit')
    return result
