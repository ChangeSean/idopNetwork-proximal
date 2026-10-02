"""Discovery-only molecular construction and independent-sample bridge inference.

All molecular design decisions, including readout coordinates and instrument
pivots, are learned on discovery patients. Estimation patients only supply the
bridge moments and their covariance. Failed designs return the whole real line.
"""
import numpy as np
from scipy.linalg import qr
from joint_readout_bridge import joint_projection
from conditional_design_bridge import select_conditional_design
import fieller_bridge as FB


def partition(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    cut = n // 2
    return order[:cut], order[cut:]


def residual_views(A, W, Z, C, P):
    n = len(A)
    C = np.zeros((n, 0)) if C is None else np.asarray(C, float)
    base = np.column_stack([np.ones(n), C])
    H = np.column_stack([A, W @ P])
    Q = np.column_stack([A, Z])
    return base, H-base @ np.linalg.lstsq(base, H, rcond=None)[0], Q-base @ np.linalg.lstsq(base, Q, rcond=None)[0]


def freeze(grid, a, treatment_eligible=None):
    row, Z, W, selected = select_conditional_design(grid, a, treatment_eligible)
    design = dict(row=dict(row), a=int(a), Z=list(Z), W=list(W), ready=False)
    if row['status'] != 'reported':
        return design
    try:
        X, C, r = selected['X'], selected['Cov'], row['r_grid']
        P = joint_projection(X, a, Z, W, r, C)
        _, H, Q = residual_views(X[:, a], X[:, W], X[:, Z], C, P)
        _, _, ix = qr((Q.T @ H[:, 1:]).T, pivoting=True, mode='economic')
        T = ix[:r].tolist()
        U = sorted(set(range(Q.shape[1]))-set(T))
        B = Q[:, T].T @ H[:, 1:] / len(X)
        sv = np.linalg.svd(B, compute_uv=False)
        if not U or sv[-1] <= max(sv[0]*1e-8, 1e-12):
            raise ValueError('Discovery nuisance coordinates unavailable')
        design.update(P=P.tolist(), pivots=dict(nuisance=T, contrasts=U), ready=True)
    except (ValueError, np.linalg.LinAlgError) as error:
        design['row'].update(status='discovery_nuisance_support', reason=str(error))
    return design


def moments(Y, A, W, Z, C, design):
    base, H, Q = residual_views(A, W, Z, C, np.asarray(design['P']))
    y = Y-base @ np.linalg.lstsq(base, Y, rcond=None)[0]
    a, R = H[:, 0], H[:, 1:]
    T, U = Q[:, design['pivots']['nuisance']], Q[:, design['pivots']['contrasts']]
    B = T.T @ R / len(A)
    sv = np.linalg.svd(B, compute_uv=False)
    if sv[-1] <= max(sv[0]*1e-8, 1e-12):
        raise ValueError('Estimation nuisance block numerically unavailable')
    L = np.linalg.solve(B.T, R.T @ U / len(A)).T
    return np.r_[U.T @ y / len(A)-L @ (T.T @ y / len(A)),
                 U.T @ a / len(A)-L @ (T.T @ a / len(A))]


def unavailable(reason, attempts=0, success=0):
    return dict(kind='all_real', intervals=[(-np.inf, np.inf)], estimate=np.nan,
                status=reason, boot_attempts=attempts, boot_success=success,
                inference='independent estimation patients; discovery design frozen')


def fit(Y, A, W, Z, C, design, draws, responses=None, alpha=.05):
    if not design['ready']:
        return unavailable('discovery_'+design['row']['status'])
    try:
        nd = moments(Y, A, W, Z, C, design)
        if not np.isfinite(nd).all():
            raise ValueError('Nonfinite observed moments')
        values = []
        for b, ix in enumerate(draws):
            try:
                yy = Y[ix] if responses is None else responses[b]
                value = moments(yy, A[ix], W[ix], Z[ix], None if C is None else C[ix], design)
                if np.isfinite(value).all():
                    values.append(value)
            except (ValueError, np.linalg.LinAlgError):
                pass
        if len(values) < max(20, .975*len(draws)):
            return unavailable('estimation_moment_resampling', len(draws), len(values))
        V = np.cov(np.asarray(values).T, ddof=1)
        result = FB.confidence_set(nd, V, alpha)
        point, derivative = FB.point_and_gradient(nd)
        result.update(estimate=point, covariance=V.tolist(),
                      normal_se=float(np.sqrt(max(derivative @ V @ derivative, 0.))),
                      status='estimated', boot_attempts=len(draws), boot_success=len(values),
                      inference='independent estimation patients; discovery design frozen')
        return result
    except (ValueError, np.linalg.LinAlgError) as error:
        result = unavailable('estimation_nuisance_support', len(draws))
        result['reason'] = str(error)
        return result
