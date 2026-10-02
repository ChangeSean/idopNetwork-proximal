"""Numerical invariants for time-integrated molecular curve decomposition."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from niche_ode import DEFAULT_DEGREE, DEFAULT_RIDGE, integrated_state_basis, ridge_refit, solve_niche_ode, decomposition_edges


def main():
    checks = []

    def check(name, actual, expected, **kwargs):
        np.testing.assert_allclose(actual, expected, **kwargs)
        checks.append(name)

    time = np.linspace(0., 1., 31)
    state = 2*time+3
    check('time integral of linear Legendre basis',
          integrated_state_basis(state[:, None], time, 1)[0][:, 0],
          time**2-time, atol=1e-12)
    X = np.column_stack([time, time**2])
    y = 2+X @ np.array([3., -4.])
    intercept, beta = ridge_refit(X, y, 0.)
    check('unpenalised polynomial recovery', np.r_[intercept, beta], [2., 3., -4.], atol=1e-12)
    base, beta = ridge_refit(X, y, 1.)
    units = np.array([.001, 1000.])
    scaled, beta_scaled = ridge_refit(X*units, y, 1.)
    check('ridge invariance to column units', base+X @ beta,
          scaled+(X*units) @ beta_scaled, atol=1e-12)
    data = pd.DataFrame({'A': state, 'B': 2+np.sin(2*time), 'C': 4.}, index=time)
    supports = pd.DataFrame([{'target': 'A', 'source': '{B}'},
                             {'target': 'B', 'source': '{}'},
                             {'target': 'C', 'source': '{}'}])
    dec = solve_niche_ode(data, supports)
    assert dec['settings']['degree'] == DEFAULT_DEGREE == 0
    assert dec['settings']['ridge'] == DEFAULT_RIDGE == .1
    zero = solve_niche_ode(data, supports, degree=0)
    base_zero, slope_zero = ridge_refit(time[:, None], data['A'].to_numpy(), DEFAULT_RIDGE)
    check('zero degree retains the common baseline', zero['predicted_states'][:, 0],
          base_zero+slope_zero[0]*np.linspace(0., 1., 50), atol=1e-12)
    check('zero degree source convention', zero['interaction_functions'][('A', 'B')], 0., atol=1e-12)
    assert decomposition_edges(zero).empty
    transformed = data.copy()
    transformed.index = 17+400*time
    other = solve_niche_ode(transformed, supports)
    check('affine niche-coordinate invariance', dec['predicted_states'],
          other['predicted_states'], atol=1e-8)
    check('constant target', dec['predicted_states'][:, 2], 4., atol=1e-10)
    for j, target in enumerate(data.columns):
        reconstructed = dec['intercepts'][j]+dec['interaction_functions'][(target, target)]
        for source in dec['support_sets'][target]:
            reconstructed = reconstructed+dec['interaction_functions'][(target, source)]
        check('additive closure '+target, reconstructed, dec['predicted_states'][:, j], atol=1e-12)
        check('centred residuals '+target, dec['diagnostics'][target]['mean_residual'], 0., atol=1e-10)
    edges = decomposition_edges(dec)
    assert set(zip(edges.source, edges.target)) <= {('B', 'A')}
    assert dec['support_sets'] == {'A': ['B'], 'B': [], 'C': []}
    checks.append('fixed-support contributions')
    result = {'passed': True, 'checks': len(checks), 'invariants': checks}
    (Path(__file__).resolve().parent / 'results/niche_ode_audit.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
