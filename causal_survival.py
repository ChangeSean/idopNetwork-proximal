"""Proximal bridges for restricted mean survival and fixed-time survival.

The censoring model estimates G(t|A,C). Conditional independence of censoring
from (T,F,Z,W) given (A,C), positivity through the horizon, and an outcome bridge
identify the intervention target. The linear mean bridge is the implementation
used here; a Cox event model is not required.
"""
import numpy as np
from proximal import proximal_bridge, ols_effect
from survival import cox_fit


def censoring_model(T, D, B=None, kind='cox'):
    T = np.asarray(T, float); D = np.asarray(D, int)
    if np.any(T < 0) or not np.isfinite(T).all(): raise ValueError('Follow-up times must be finite and nonnegative')
    if not np.isin(D, [0, 1]).all(): raise ValueError('Event indicators must be zero or one')
    n = len(T)
    if kind not in ('cox', 'km'): raise ValueError('Censoring model must be cox or km')
    if B is None: B = np.zeros((n, 0))
    B = np.asarray(B, float).reshape(n, -1)
    mean = B.mean(0); scale = B.std(0); keep = scale > 1e-10
    x = (B[:, keep] - mean[keep]) / scale[keep]
    if kind == 'cox' and x.shape[1] and np.sum(1-D):
        beta, _ = cox_fit(T, 1-D, x)
        if not np.isfinite(beta).all(): raise ValueError('Censoring model did not produce finite coefficients')
        eta = x @ beta
        if np.max(np.abs(eta)) > 35: raise ValueError('Censoring model has an unbounded linear predictor')
    else:
        beta = np.zeros(x.shape[1]); eta = np.zeros(n)
    times, counts = np.unique(T[D == 0], return_counts=True)
    if not len(times):
        return dict(times=times, cumulative=np.array([]), eta=eta, kind=kind, beta=beta)
    order = np.argsort(T); ts = T[order]
    weighted_tail = np.r_[np.cumsum(np.exp(eta[order])[::-1])[::-1], 0.]
    risk = weighted_tail[np.searchsorted(ts, times, side='left')]
    if kind == 'km':
        survival = np.cumprod(1-counts/risk)
        cumulative = -np.log(np.maximum(survival, 1e-300))
    else:
        cumulative = np.cumsum(counts / risk)
    return dict(times=times, cumulative=cumulative, eta=eta, kind=kind, beta=beta)


def censor_survival(model, times):
    times = np.asarray(times, float)
    indices = np.searchsorted(model['times'], times, side='right')
    cumulative = np.r_[0., model['cumulative']][indices]
    return np.exp(-np.exp(model['eta'])[:, None] * cumulative[None, :])


def survival_pseudo_outcomes(T, D, horizon, B=None, censor='cox', g_min=.05):
    """Integral IPCW RMST and indicator IPCW survival at the same horizon.

    The positivity floor is checked rather than used to truncate weights.
    RMST integration is exact for the step baseline censoring estimator.
    """
    T = np.asarray(T, float); D = np.asarray(D, int)
    if horizon <= 0: raise ValueError('Horizon must be positive')
    model = censoring_model(T, D, B, censor)
    g_horizon = censor_survival(model, [horizon])[:, 0]
    if np.any(g_horizon < g_min): raise ValueError('Censoring positivity is below the reporting floor')
    nodes = np.unique(np.r_[0., model['times'][(model['times'] > 0) & (model['times'] < horizon)], horizon])
    left, right = nodes[:-1], nodes[1:]
    if np.all(model['eta'] == 0):
        # A common censoring curve admits one cumulative integral for all patients.
        g_left = censor_survival({**model, 'eta': np.zeros(1)}, left)[0]
        cumulative_area = np.r_[0., np.cumsum((right-left)/g_left)]
        end = np.minimum(T, horizon)
        interval = np.minimum(np.searchsorted(nodes, end, side='right')-1, len(left)-1)
        rmst = cumulative_area[interval] + (end-left[interval])/g_left[interval]
    else:
        indices = np.searchsorted(model['times'], left, side='right')
        hazard_left = np.r_[0., model['cumulative']][indices]
        rmst = np.empty(len(T))
        for start in range(0, len(T), 256):
            end = min(start+256, len(T))
            durations = np.maximum(np.minimum(T[start:end,None], right[None,:])-left[None,:], 0.)
            rmst[start:end] = np.sum(durations*np.exp(np.exp(model['eta'][start:end,None])*hazard_left[None,:]),axis=1)
    survival = (T > horizon).astype(float) / g_horizon
    return dict(rmst=rmst, survival=survival, min_g=float(g_horizon.min()),
                median_g=float(np.median(g_horizon)), horizon=horizon)


def fit_survival_bridge(T, D, A, W, Z, rank, C=None, horizon=36., censor='cox',
                        n_boot=300, seed=20260930, g_min=.05, include_naive=True):
    """Fixed selected roles, censoring and both bridge stages refitted per draw."""
    n = len(A); A = np.asarray(A, float)
    W = np.asarray(W, float).reshape(n, -1); Z = np.asarray(Z, float).reshape(n, -1)
    C = np.zeros((n, 0)) if C is None else np.asarray(C, float).reshape(n, -1)
    B = np.column_stack([A, C])
    pseudo = survival_pseudo_outcomes(T, D, horizon, B, censor, g_min)
    estimates = {}
    for target in ('rmst', 'survival'):
        value, _, nu = proximal_bridge(pseudo[target], A, W, Z, rank, C, min_sv_ratio=1e-8)
        naive, _ = ols_effect(pseudo[target], A, C)
        estimates[target] = dict(estimate=float(value), naive=float(naive), nu=float(nu))
    draws = {target: [] for target in estimates}; naive_draws = {target: [] for target in estimates}
    rng = np.random.default_rng(seed); failed = 0
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        try:
            bp = survival_pseudo_outcomes(np.asarray(T)[ix], np.asarray(D)[ix], horizon, B[ix], censor, g_min)
            values = {}; naive_values = {}
            for target in estimates:
                values[target] = proximal_bridge(bp[target], A[ix], W[ix], Z[ix], rank, C[ix], min_sv_ratio=1e-8)[0]
                if include_naive: naive_values[target] = ols_effect(bp[target], A[ix], C[ix])[0]
            if not np.isfinite(list(values.values())).all(): raise ValueError('Nonfinite bridge fit')
            for target in estimates:
                draws[target].append(values[target])
                if include_naive: naive_draws[target].append(naive_values[target])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            failed += 1
    for target in estimates:
        enough = len(draws[target]) >= max(20, int(.9*n_boot))
        se = float(np.std(draws[target], ddof=1)) if enough else np.nan
        naive_se = float(np.std(naive_draws[target], ddof=1)) if enough and include_naive else np.nan
        estimates[target].update(se_boot=se, naive_se_boot=naive_se, boot_success=len(draws[target]),
                                 boot_failed=failed, ci_low=estimates[target]['estimate']-1.96*se,
                                 ci_high=estimates[target]['estimate']+1.96*se)
    return dict(targets=estimates, min_g=pseudo['min_g'], median_g=pseudo['median_g'], rank=rank)
