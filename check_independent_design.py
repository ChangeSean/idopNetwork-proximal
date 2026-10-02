"""Independent-design checks: held-out isolation, bridge closure and fallback."""
import copy
import json
from pathlib import Path
import numpy as np
import independent_design_bridge as ID
import fieller_bridge as FB


def main():
    rng = np.random.default_rng(20261401)
    n = 16000
    F = rng.normal(size=(n, 2))
    A = F @ np.array([.8, .6])+rng.normal(size=n)
    Z = F+.35*rng.normal(size=(n, 2))
    W = F+.35*rng.normal(size=(n, 2))
    Y = .5*A+F @ np.array([1., 1.2])+rng.normal(size=n)
    X = np.column_stack([A, Z, W])
    d, e = ID.partition(n, 20261402)
    assert not set(d)&set(e) and len(set(np.r_[d,e])) == n
    graph = np.zeros((5, 5), bool)
    graph[0, [1, 2]] = True
    graph[[1, 2], 0] = True
    grid = [dict(X=X[d], Cov=np.zeros((len(d), 0)), names=[f'v{i}' for i in range(5)],
                 Und=graph, alpha=.15, latent=dict(r=2, Lam=np.ones((5, 2))))]
    design = ID.freeze(grid, 0)
    assert design['ready'] and design['row']['r_grid'] == 2
    snapshot = json.dumps(design, sort_keys=True)
    nd = ID.moments(Y[e], A[e], W[e], Z[e], None, design)
    assert abs(FB.point_and_gradient(nd)[0]-.5) < .06
    shifted = ID.moments(Y[e]+8, A[e]+3, W[e]+2, Z[e]-7, None, design)
    assert np.allclose(nd, shifted, atol=1e-10)
    rotated = copy.deepcopy(design)
    rotation = np.array([[0., -1.], [1., 0.]])
    rotated['P'] = (np.asarray(design['P']) @ rotation).tolist()
    assert np.allclose(nd, ID.moments(Y[e], A[e], W[e], Z[e], None, rotated), atol=1e-10)
    rescaled = ID.moments(Y[e], 4*A[e], W[e], Z[e], None, design)
    assert abs(FB.point_and_gradient(rescaled)[0]*4-FB.point_and_gradient(nd)[0]) < 1e-10
    # Estimation data/outcomes cannot change a serialised discovery design.
    ID.moments(-Y[e], A[e], W[e], Z[e], None, design)
    assert json.dumps(design, sort_keys=True) == snapshot
    draws = [rng.integers(0, len(e), len(e)) for _ in range(100)]
    result = ID.fit(Y[e], A[e], W[e], Z[e], None, design, draws)
    assert result['status'] == 'estimated' and result['boot_success'] == 100
    assert FB.contains(result, .5)
    failed = ID.fit(Y[e], A[e], W[e], Z[e], None, dict(design, ready=False), draws)
    assert failed['kind'] == 'all_real' and np.isnan(failed['estimate'])
    response = [Y[e][ix] for ix in draws]
    response[:3] = [np.full(len(e), np.nan) for _ in range(3)]
    incomplete = ID.fit(Y[e], A[e], W[e], Z[e], None, design, draws, response)
    assert incomplete['status'] == 'estimation_moment_resampling' and incomplete['boot_success'] == 97
    response[2] = Y[e][draws[2]]
    complete = ID.fit(Y[e], A[e], W[e], Z[e], None, design, draws, response)
    assert complete['status'] == 'estimated' and complete['boot_success'] == 98
    path = Path(__file__).parent/'results/independent_design_validation_20261001'
    path.mkdir(parents=True, exist_ok=True)
    (path/'algebra_audit.json').write_text(json.dumps(dict(passed=True, checks=12,
             effect=result['estimate'], truth=.5, estimate_patients=len(e),
             design_frozen=True), indent=2))
    print('Independent bridge: 12 checks passed; effect', result['estimate'])


if __name__ == '__main__':
    main()
