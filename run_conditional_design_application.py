"""Outcome-blind conditional design and complete molecular construction bootstrap.

Rebuild curves, standardisation, niche ordering, eight support graphs, proxy
sets, rank, design selection, projection, censoring and both bridge stages.
The declared protein panel and initial clinical covariate imputations are fixed.
All failed constructions are recorded; no finite CI for incomplete pipelines.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from run_application import load_cohort
from multiscale_bridge import design_grid
from conditional_design_bridge import select_conditional_design, bootstrap_interval
from joint_readout_bridge import estimate
from causal_survival import survival_pseudo_outcomes
from diagnose_construction import resample_cohort
from proximal import bh

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/conditional_design_application_20261001'


def run(study, boots=300):
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(study, 'OS', survival=True)
    grid = design_grid(cohort)
    X, C, names = grid[0]['X'], grid[0]['Cov'], cohort['names']
    start = time.time()
    rows, chosen = [], {}
    for a in range(len(names)):
        row, Z, W, res = select_conditional_design(grid, a)
        row = dict(row, n=len(X), events=int(cohort['Y'].sum()), horizon=36)
        if row['status'] == 'reported':
            try:
                py = survival_pseudo_outcomes(cohort['T'], cohort['Y'], 36, np.column_stack([X[:, a], C]))
                row['min_g'] = py['min_g']
                for target in ('rmst', 'survival'):
                    row[target+'_estimate'] = estimate(py[target], X[:, a], X[:, W], X[:, Z], row['r_bridge'], C)
                chosen[a] = (row, Z, W)
            except (ValueError, np.linalg.LinAlgError) as error:
                row.update(status='outcome_support', reason=str(error))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/f'{study}_point_designs.csv', index=False)
    print(study, 'point designs', pd.DataFrame(rows).status.value_counts().to_dict(), flush=True)
    records, values = [], {a: {'rmst': [], 'survival': []} for a in chosen}
    rng = np.random.default_rng(20261011)
    for b in range(boots):
        ix = rng.integers(0, len(X), len(X))
        try:
            sampled, ordered = resample_cohort(cohort, X, ix)
            new_grid = design_grid(sampled)
            xb, cb = new_grid[0]['X'], new_grid[0]['Cov']
            scale = X[ix].std(0, ddof=0)  # original-SD target, not bootstrap-SD target
            for a in chosen:
                row, Z, W, res = select_conditional_design(new_grid, a)
                record = dict(row, replicate=b, exposure_scale=scale[a])
                if row['status'] == 'reported':
                    try:
                        py = survival_pseudo_outcomes(sampled['T'], sampled['Y'], 36,
                                                      np.column_stack([xb[:, a], cb]))
                        for target in ('rmst', 'survival'):
                            record[target+'_estimate'] = estimate(py[target], xb[:, a], xb[:, W], xb[:, Z],
                                                                  row['r_bridge'], cb)/scale[a]
                        if not np.isfinite([record['rmst_estimate'], record['survival_estimate']]).all():
                            raise ValueError('Nonfinite complete-pipeline estimate')
                        record['min_g'] = py['min_g']
                        for target in values[a]:
                            values[a][target].append(record[target+'_estimate'])
                    except (ValueError, np.linalg.LinAlgError) as error:
                        record.update(status='outcome_support', reason=str(error))
                records.append(record)
        except (ValueError, np.linalg.LinAlgError) as error:
            for a in chosen:
                records.append(dict(replicate=b, exposure=names[a], status='construction_failure', reason=str(error)))
        if (b+1) % 20 == 0 or b+1 == boots:
            pd.DataFrame(records).to_csv(OUT/f'{study}_full_pipeline_replicates.csv', index=False)
            print(study, 'full pipeline', b+1, '/', boots, 'seconds', round(time.time()-start, 1), flush=True)
    for a, (row, Z, W) in chosen.items():
        for target in values[a]:
            ci = bootstrap_interval(row[target+'_estimate'], values[a][target], boots)
            row.update({target+'_'+key: value for key, value in ci.items()})
        row['status'] = 'reported' if np.isfinite(row['rmst_se_boot']) else 'construction_incomplete'
    result = pd.DataFrame(rows)
    for target in ('rmst', 'survival'):
        key = target+'_se_boot'
        if key in result:
            good = result.status.eq('reported') & result[key].gt(0)
            result.loc[good, target+'_p'] = 2*norm.sf(abs(result.loc[good, target+'_estimate']/result.loc[good, key]))
            result.loc[good, target+'_q'] = bh(result.loc[good, target+'_p'].to_numpy())
    result.to_csv(OUT/f'{study}_full_pipeline_results.csv', index=False)
    pd.DataFrame(records).to_csv(OUT/f'{study}_full_pipeline_replicates.csv', index=False)
    manifest = dict(study=study, boots=boots, seed=20261011,
                    selection='max smallest conditional canonical root among full joint/conditional rank designs',
                    resampling='all molecular construction and censoring stages; original exposure-SD units',
                    fixed='initial 60-protein panel, initial molecular and covariate imputations',
                    interval='normal bootstrap; >=97.5% complete attempts and >=20 draws',
                    validity='pointwise stable identified model, not uniform weak-identification inference',
                    elapsed_seconds=time.time()-start,
                    source_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                                   ('conditional_design_bridge.py', 'run_conditional_design_application.py',
                                    'joint_readout_bridge.py', 'multiscale_bridge.py')})
    (OUT/f'{study}_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(study, 'finished', result.status.value_counts().to_dict(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('study', choices=['ov', 'luad'])
    p.add_argument('--boot', type=int, default=300)
    args = p.parse_args()
    run(args.study, args.boot)
