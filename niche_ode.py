"""Time-integrated niche ODE decomposition with standardised ridge refitting.

Uses the smoothing, time integration and grouped contribution construction in
the supplied MTODE reference. The existing molecular support supplies groups;
the clinical graph is not reselected by a separate ADSIHT screening procedure.
"""
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import make_smoothing_spline
from scipy.special import eval_legendre

DEFAULT_RIDGE = .1


def integrated_state_basis(states, time, degree):
    """Integrate state basis functions along time, rather than state changes."""
    states = np.asarray(states, float)
    time = np.asarray(time, float)
    if states.ndim != 2 or len(states) != len(time) or np.any(np.diff(time) <= 0):
        raise ValueError('Basis requires a matrix on a strictly increasing time grid')
    if degree == 0:
        return [np.empty((len(time), 0)) for _ in states.T]
    blocks = []
    for values in states.T:
        span = np.ptp(values)
        if span <= 1e-12 * max(np.max(np.abs(values)), 1.):
            blocks.append(np.zeros((len(time), degree)))
            continue
        scaled = 2 * (values-values.min()) / span - 1
        phi = np.column_stack([eval_legendre(k, scaled) for k in range(1, degree+1)])
        blocks.append(cumulative_trapezoid(phi, time, axis=0, initial=0))
    return blocks


def ridge_refit(X, y, ridge):
    """Centre and standardise columns; recover original-scale coefficients."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    centre = X.mean(axis=0)
    scale = X.std(axis=0, ddof=1)
    active = scale > 1e-12 * np.maximum(np.max(np.abs(X), axis=0), 1.)
    beta = np.zeros(X.shape[1])
    if active.any():
        design = (X[:, active]-centre[active]) / scale[active]
        u, singular, vt = np.linalg.svd(design, full_matrices=False)
        if ridge == 0:
            coef = np.linalg.lstsq(design, y-y.mean(), rcond=1e-12)[0]
        else:
            coef = vt.T @ ((singular/(singular**2+ridge)) * (u.T @ (y-y.mean())))
        beta[active] = coef / scale[active]
    intercept = float(y.mean()-centre @ beta)
    return intercept, beta


def solve_niche_ode(data, supports, degree=1, ridge=DEFAULT_RIDGE, n_grid=50):
    """Smooth raw states, fit fixed source groups, and return additive curves.

    Each target has a baseline time column plus its own and supported sources'
    degree-1,...,degree shifted Legendre bases. The constant basis is represented
    once by time. Intercept and time baseline belong to the intrinsic component.
    Degree zero retains only that shared constant-derivative baseline; source
    groups have no additional columns and their reported contributions are zero.
    The ridge penalty uses the unnormalised sum-of-squares convention.
    """
    if degree < 0 or int(degree) != degree or ridge < 0 or not np.isfinite(ridge):
        raise ValueError('Use nonnegative integer degree and finite nonnegative ridge')
    degree = int(degree)
    if n_grid < 5 or not isinstance(data, pd.DataFrame):
        raise ValueError('Use a DataFrame and at least five output grid points')
    if not data.columns.is_unique:
        raise ValueError('Protein names must be unique')
    features = list(data.columns)
    s, values = np.asarray(data.index, float), data.to_numpy(dtype=float)
    if not np.isfinite(s).all() or not np.isfinite(values).all():
        raise ValueError('Niche coordinate and protein values must be finite')
    order = np.argsort(s, kind='stable')
    s, values = s[order], values[order]
    unique, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    if len(unique) < 5 or unique[-1] <= unique[0]:
        raise ValueError('At least five distinct niche coordinates are needed')
    averaged = np.zeros((len(unique), len(features)))
    np.add.at(averaged, inverse, values)
    averaged /= counts[:, None]
    duration = unique[-1]-unique[0]
    time_unique = (unique-unique[0]) / duration
    time_observed = (s-unique[0]) / duration
    time_grid = np.linspace(0., 1., n_grid)
    # One integration grid ensures consistent training and plotting integrals.
    time_union = np.unique(np.r_[time_unique, time_grid])
    if degree:
        smoothed = np.column_stack([
            make_smoothing_spline(time_unique, averaged[:, j], w=counts.astype(float))(time_union)
            for j in range(len(features))])
        blocks = integrated_state_basis(smoothed, time_union, degree)
        blocks_observed = [np.column_stack([np.interp(time_observed, time_union, b[:, k])
                                          for k in range(degree)]) for b in blocks]
        blocks_grid = [np.column_stack([np.interp(time_grid, time_union, b[:, k])
                                      for k in range(degree)]) for b in blocks]
    else:
        blocks_observed = [np.empty((len(s), 0)) for _ in features]
        blocks_grid = [np.empty((n_grid, 0)) for _ in features]
    support_sets = {target: [] for target in features}
    for row in supports.itertuples():
        if row.target not in support_sets:
            raise ValueError('Unknown target in molecular support: ' + row.target)
        sources = str(row.source).strip('{}').split(',')
        for source in sources:
            source = source.strip()
            if not source:
                continue
            if source not in features:
                raise ValueError('Unknown source in molecular support: ' + source)
            if source != row.target and source not in support_sets[row.target]:
                support_sets[row.target].append(source)
    interaction_functions = {}
    intercepts = np.zeros(len(features))
    prediction = np.zeros((n_grid, len(features)))
    fitted_observed = np.zeros_like(values)
    diagnostics = {}
    for j, target in enumerate(features):
        sources = [target] + support_sets[target]
        ix = [features.index(source) for source in sources]
        X = np.column_stack([time_observed] + [blocks_observed[k] for k in ix])
        intercept, beta = ridge_refit(X, values[:, j], ridge)
        intercepts[j] = intercept
        fitted_observed[:, j] = intercept + X @ beta
        reconstructed = np.full(n_grid, intercept) + beta[0] * time_grid
        for g, (source, k) in enumerate(zip(sources, ix)):
            effect = blocks_grid[k] @ beta[1+g*degree:1+(g+1)*degree]
            if source == target:
                interaction_functions[(target, source)] = effect + beta[0] * time_grid
            else:
                interaction_functions[(target, source)] = effect
            reconstructed += effect
        prediction[:, j] = reconstructed
        intrinsic = intercept + interaction_functions[(target, target)]
        cross = [interaction_functions[(target, source)] for source in support_sets[target]]
        closure = intrinsic + sum(cross, np.zeros(n_grid))
        closure_error = float(np.max(np.abs(reconstructed-closure)))
        assert closure_error < 1e-9 * max(np.max(np.abs(reconstructed)), 1.)
        residual = values[:, j] - fitted_observed[:, j]
        total_ss = np.sum((values[:, j]-values[:, j].mean())**2)
        intrinsic_rms = float(np.sqrt(np.mean(intrinsic**2)))
        cross_rms = [float(np.sqrt(np.mean(c**2))) for c in cross]
        # Variation measures remove the arbitrary abundance offset from comparison.
        variation_rms = lambda curve: float(np.sqrt(np.mean((curve-curve.mean())**2)))
        intrinsic_variation = variation_rms(intrinsic)
        cross_variation = [variation_rms(c) for c in cross]
        predicted_variation = variation_rms(reconstructed)
        scale = X.std(axis=0, ddof=1)
        active = scale > 1e-12 * np.maximum(np.max(np.abs(X), axis=0), 1.)
        singular = np.linalg.svd((X[:, active]-X[:, active].mean(axis=0))/scale[active],
                                 compute_uv=False) if active.any() else np.array([])
        effective_df = 1 + (float(np.sum(singular**2/(singular**2+ridge))) if ridge else
                            float(np.sum(singular > (singular[0]*1e-12 if len(singular) else 0))))
        gcv = float(np.mean(residual**2)/(1-effective_df/len(values))**2)
        diagnostics[target] = dict(rmse=float(np.sqrt(np.mean(residual**2))),
                                   r_squared=float(1-np.sum(residual**2)/total_ss) if total_ss else None,
                                   mean_residual=float(residual.mean()), closure_error=closure_error,
                                   intrinsic_rms=intrinsic_rms, max_cross_rms=max(cross_rms, default=0.),
                                   max_cross_to_intrinsic_rms=max(cross_rms, default=0.)/max(intrinsic_rms, 1e-12),
                                   intrinsic_variation_rms=intrinsic_variation,
                                   max_cross_variation_rms=max(cross_variation, default=0.),
                                   cancellation_index=(intrinsic_variation+sum(cross_variation))/max(predicted_variation, 1e-12),
                                   ridge_effective_df=effective_df, fixed_design_gcv=gcv,
                                   source_groups=len(cross), coefficients=beta.tolist())
    return dict(features=features, sample_tau=unique[0]+time_grid*duration,
                intercepts=intercepts, predicted_states=prediction,
                interaction_functions=interaction_functions, support_sets=support_sets,
                fitted_observed=fitted_observed, diagnostics=diagnostics,
                settings=dict(smoothing='GCV cubic smoothing spline' if degree else 'not needed for constant basis',
                              basis='shifted Legendre',
                              degree=degree, ridge=ridge, integration='normalised niche time',
                              grid_points=n_grid, selection='fixed molecular support',
                              intrinsic='intercept + linear time baseline + own state group'))


def decomposition_edges(decomposition):
    """MTODE-style edge weights: mean cumulative source contribution."""
    rows = []
    for target in decomposition['features']:
        for source in decomposition['support_sets'][target]:
            effect = decomposition['interaction_functions'][(target, source)]
            if np.max(np.abs(effect)) > 1e-10:
                rows.append(dict(source=source, target=target, weight=float(np.mean(effect))))
    return pd.DataFrame(rows, columns=['source', 'target', 'weight'])
