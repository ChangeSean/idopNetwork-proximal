"""Prespecified window-width prototype, separate from the primary estimator.

Five overlapping niche windows each cover 40% of patients (twice the original
20%); starts are equally spaced. Penalties and the >60% support rule stay fixed.
This prototype is evaluated on design information before any clinical outcomes.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from proximal import components
from run_application import load_cohort
from run_causal_application import analyse_design_grid, configuration_grid
from diagnose_construction import OUT, resample_cohort


def overlapping_grid(grid, fraction=.4):
    result = []
    X = grid[0]['X']; n, p = X.shape
    width = int(np.ceil(n*fraction))
    starts = np.rint(np.linspace(0, n-width, 5)).astype(int)
    for reference in grid:
        count = np.zeros((p, p), dtype=int)
        for start in starts:
            window = X[start:start+width]
            sd = window.std(0); sd[sd == 0] = 1
            standard = (window-window.mean(0))/sd
            for a in range(p):
                others = np.array([j for j in range(p) if j != a])
                model = Lasso(alpha=reference['alpha'], fit_intercept=True, max_iter=100000, tol=1e-5)
                model.fit(standard[:, others], standard[:, a])
                count[others, a] += model.coef_ != 0
        A = count/5 > .6; Und = A | A.T
        result.append(dict(reference, A=A, Und=Und, ncomp=len(set(components(Und))),
                           window_fraction=fraction))
    return result


def run(study, reps):
    OUT.mkdir(exist_ok=True, parents=True)
    cohort = load_cohort(study, 'OS', survival=True)
    reference = analyse_design_grid(cohort); n = len(reference[0]['X'])
    rows = []; rng = np.random.default_rng(20261001); started=time.time()
    for b in range(-1, reps):
        grid = reference if b == -1 else analyse_design_grid(resample_cohort(
            cohort, reference[0]['X'], rng.integers(0, n, n))[0])
        for method, designs in [('nonoverlap', grid), ('overlap40', overlapping_grid(grid))]:
            for a in range(len(cohort['names'])):
                row, Z, W, chosen = configuration_grid(designs, a)
                rows.append(dict(row, study=study, replicate=b, method=method,
                                 components=chosen['ncomp'], directed_edges=int(chosen['A'].sum())))
        if b % 5 == 4: print(study, b+1, 'elapsed', round(time.time()-started, 1), flush=True)
    table=pd.DataFrame(rows); table.to_csv(OUT/f'{study}_overlap_prototype.csv', index=False)
    print(table.assign(reference=table.replicate.eq(-1)).groupby(['reference','method']).status.value_counts().to_string(),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('study', choices=['ov','luad'])
    parser.add_argument('--reps',type=int,default=30);args=parser.parse_args()
    run(args.study,args.reps)
