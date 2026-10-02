"""Outcome-blind decomposition of molecular construction resampling.

Keeps published results intact. Crossed graph/moment calculations diagnose
sources of variation; they are not independent validation or effect estimates.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from idop_core import data_transformation, data_quasi_dynamic
from run_application import load_cohort
from run_causal_application import analyse_design_grid, configuration, configuration_grid
from proximal import bridge_information_rank, proxy_strength

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'construction_diagnosis_20261001'


def resample_cohort(cohort, X, indices):
    scaled = data_transformation(pd.DataFrame(X[indices], columns=cohort['names']), 'Zscore_shift')
    order = np.argsort(scaled.sum(axis=1).to_numpy(), kind='stable')
    selected = indices[order]
    return dict(qd=data_quasi_dynamic(scaled), names=cohort['names'],
                Y=cohort['Y'][selected], Cov=cohort['Cov'][selected], T=cohort['T'][selected]), selected


def crossed_grid(graphs, moments):
    return [{**graph, 'X': moments['X'], 'Cov': moments['Cov']} for graph in graphs]


def describe(grid, a):
    chosen, Z, W, res = configuration_grid(grid, a)
    row = dict(chosen)
    row['components'] = res['ncomp']
    candidates = []
    for item in grid:
        local, zz, ww = configuration(item, a)
        local = dict(local, alpha=item['alpha'], components=item['ncomp'],
                     directed_edges=int(item['A'].sum()), r_grid=row['r_grid'])
        if zz and ww:
            joint = bridge_information_rank(item['X'][:, ww], item['X'][:, [a]+zz], None, item['Cov'])
            conditional = bridge_information_rank(item['X'][:, ww], item['X'][:, zz], item['X'][:, a], item['Cov'])
            local.update(joint_roots=';'.join(map(str, joint['roots'])),
                         joint_pvalues=';'.join(map(str, joint['pvalues'])),
                         conditional_pvalues=';'.join(map(str, conditional['pvalues'])),
                         joint_full_rank=joint['r'] == min(len(ww), len(zz)+1),
                         conditional_full_rank=conditional['r'] == min(len(ww), len(zz)))
        candidates.append(local)
    row['local_passes'] = sum(c['status'] == 'reported' for c in candidates)
    row['grid_blocks_local_pass'] = row['status'] != 'reported' and row['local_passes'] > 0
    row['joint_exceeds_z_count'] = row['r_grid'] > row['nZ']
    return row, candidates


def run(study, reps=60):
    OUT.mkdir(exist_ok=True, parents=True)
    cohort = load_cohort(study, 'OS', survival=True)
    reference = analyse_design_grid(cohort)
    X = reference[0]['X']; n = len(X)
    clinical = pd.read_csv(ROOT/'results'/f'{study}_OS_causal_survival.csv')
    focus = set(clinical.loc[clinical.status.eq('reported'), 'exposure'])
    rng = np.random.default_rng(20261001)
    subsample_rng = np.random.default_rng(20261002)
    rows, candidates, windows, fixed = [], [], [], []
    started = time.time()
    reference_roles = {a: configuration_grid(reference, a) for a in range(len(cohort['names']))}
    for a, name in enumerate(cohort['names']):
        row, detail = describe(reference, a)
        rows.append(dict(row, study=study, replicate=-1, mode='reference', focus=name in focus))
        candidates.extend(dict(c, study=study, replicate=-1, mode='reference', focus=name in focus) for c in detail)
    for b in range(reps):
        ix = rng.integers(0, n, n)
        bootstrap, ordered_ix = resample_cohort(cohort, X, ix)
        rebuilt = analyse_design_grid(bootstrap)
        subix = subsample_rng.choice(n, size=int(np.floor(.8*n)), replace=False)
        subcohort, ordered_subix = resample_cohort(cohort, X, subix)
        subgrid = analyse_design_grid(subcohort)
        modes = {'bootstrap_full': rebuilt,
                 'bootstrap_fixed_graph': crossed_grid(reference, rebuilt[0]),
                 'bootstrap_graph_original_moments': crossed_grid(rebuilt, reference[0]),
                 'subsample80_full': subgrid}
        for mode, identifiers in [('bootstrap_full', ordered_ix), ('subsample80_full', ordered_subix)]:
            for w, ids in enumerate(np.array_split(identifiers, 5)):
                windows.append(dict(study=study, replicate=b, mode=mode, window=w,
                                    rows=len(ids), unique_patients=len(np.unique(ids))))
        for a, name in enumerate(cohort['names']):
            for mode, grid in modes.items():
                row, detail = describe(grid, a)
                rows.append(dict(row, study=study, replicate=b, mode=mode, focus=name in focus))
                candidates.extend(dict(c, study=study, replicate=b, mode=mode, focus=name in focus) for c in detail)
            # Hold both original sets and original dimension fixed to locate
            # whether variation is due to reselecting rank or weak moments.
            original, Z, W, _ = reference_roles[a]
            if name in focus:
                for mode, grid in [('bootstrap_fixed_roles', rebuilt), ('subsample80_fixed_roles', subgrid)]:
                    xx, cc = grid[0]['X'], grid[0]['Cov']
                    joint = bridge_information_rank(xx[:, W], xx[:, [a]+Z], None, cc)
                    cond = bridge_information_rank(xx[:, W], xx[:, Z], xx[:, a], cc)
                    strength = proxy_strength(xx[:, W], xx[:, Z], xx[:, a], original['r_bridge'], cc)
                    fixed.append(dict(study=study, replicate=b, exposure=name, mode=mode,
                                      r_reference=original['r_bridge'], r_joint=joint['r'], r_signal=cond['r'],
                                      nu_fixed_rank=strength, fixed_rank_strength_pass=strength >= .05,
                                      reranked_pass=joint['r'] == cond['r'] and joint['r'] > 0 and
                                      proxy_strength(xx[:, W], xx[:, Z], xx[:, a], joint['r'], cc) >= .05))
        if (b+1) % 5 == 0 or b+1 == reps:
            print(study, b+1, '/', reps, 'elapsed', round(time.time()-started, 1), flush=True)
    for suffix, records in [('designs', rows), ('candidates', candidates), ('windows', windows), ('fixed_roles', fixed)]:
        pd.DataFrame(records).to_csv(OUT/f'{study}_{suffix}.csv', index=False)
    # Exact replay verifies patient ordering and diagnostic implementation.
    saved = pd.read_csv(ROOT/'results'/f'{study}_OS_proxy_construction_replicates.csv')
    replay = pd.DataFrame(rows).query("mode == 'bootstrap_full'")
    merged = replay.merge(saved, on=['replicate', 'exposure'], suffixes=('_new', '_old'))
    checks = {}
    for col in ['status', 'Z', 'W', 'r_grid', 'r_bridge', 'alpha', 'components']:
        checks[col] = bool(merged[col+'_new'].fillna('').eq(merged[col+'_old'].fillna('')).all())
    assert len(merged) == min(reps, 60)*len(cohort['names'])
    assert all(checks.values()), checks
    metadata = dict(study=study, reps=reps, seed_bootstrap=20261001, seed_subsample=20261002,
                    subsample_fraction=.8, matched_rows=len(merged), exact_replay_checks=checks,
                    scope='Fixed 60-protein panel and imputation; conditional on initial panel preprocessing. No outcome used in design decisions.',
                    crossed_modes='Diagnostic counterfactual calculations on the same cohort; not independent validation.',
                    elapsed_seconds=time.time()-started)
    (OUT/f'{study}_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study', choices=['ov', 'luad'])
    parser.add_argument('--reps', type=int, default=60)
    args = parser.parse_args()
    run(args.study, args.reps)
