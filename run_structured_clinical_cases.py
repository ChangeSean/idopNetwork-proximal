"""Apply an explicit biological boundary before inspecting worked-case effects."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_application import load_cohort
from multiscale_bridge import design_grid
from conditional_design_bridge import select_conditional_design
from run_joint_readout_application import fit
from diagnose_construction import resample_cohort
from causal_survival import survival_pseudo_outcomes
from joint_readout_bridge import estimate

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/conditional_design_application_20261001'


def apply_boundary(grid, names, case):
    blocked = set(case['exclude_from_both_roles'])
    eligible_z = np.array([name not in blocked|set(case['exclude_from_treatment']) for name in names])
    masked = []
    for res in grid:
        loading = res['latent']['Lam'].copy()
        loading[[j for j, name in enumerate(names) if name in blocked]] = 0.
        masked.append(dict(res, latent=dict(res['latent'], Lam=loading)))
    return masked, eligible_z


def construction_case(boots=300):
    cases = json.loads((ROOT/'clinical_case_boundaries_20261001.json').read_text(encoding='utf-8'))['cases']
    records = []
    for case in cases:
        if case['exposure'] != 'KDR':
            continue  # GAB2 has no point design under its specified boundary.
        c = load_cohort(case['study'], 'OS', survival=True)
        grid = design_grid(c)
        X, names = grid[0]['X'], c['names']
        a = names.index(case['exposure'])
        rng = np.random.default_rng(20261041)
        for b in range(boots):
            ix = rng.integers(0, len(X), len(X))
            sampled, _ = resample_cohort(c, X, ix)
            g = design_grid(sampled)
            ref, _, _, _ = select_conditional_design(g, a)
            masked, eligible_z = apply_boundary(g, names, case)
            row, Z, W, chosen = select_conditional_design(masked, a, eligible_z)
            row = dict(row, replicate=b, reference_joint_dimension=ref['r_grid'])
            if row['r_grid'] < ref['r_grid']:
                row['status'] = 'boundary_readout_dimension'
            if row['status'] == 'reported':
                try:
                    xb, cb = chosen['X'], chosen['Cov']
                    py = survival_pseudo_outcomes(sampled['T'], sampled['Y'], 36, np.column_stack([xb[:, a], cb]))
                    for target in ('rmst', 'survival'):
                        row[target+'_estimate'] = estimate(py[target], xb[:, a], xb[:, W], xb[:, Z], row['r_grid'], cb)/X[ix, a].std()
                except (ValueError, np.linalg.LinAlgError) as error:
                    row.update(status='outcome_support', reason=str(error))
            records.append(row)
            if (b+1) % 20 == 0:
                pd.DataFrame(records).to_csv(OUT/'structured_case_construction.csv', index=False)
                print('KDR structured construction', b+1, '/', boots, flush=True)
    pd.DataFrame(records).to_csv(OUT/'structured_case_construction.csv', index=False)


def main():
    protocol = json.loads((ROOT/'clinical_case_boundaries_20261001.json').read_text(encoding='utf-8'))
    records = []
    for case in protocol['cases']:
        c = load_cohort(case['study'], 'OS', survival=True)
        grid = design_grid(c)
        names = c['names']
        a = names.index(case['exposure'])
        reference, _, _, _ = select_conditional_design(grid, a)
        blocked = set(case['exclude_from_both_roles'])
        masked, eligible_z = apply_boundary(grid, names, case)
        row, Z, W, res = select_conditional_design(masked, a, eligible_z)
        row = dict(row, study=case['study'], boundary=case['boundary'],
                   reference_joint_dimension=reference['r_grid'],
                   excluded_measured=';'.join(name for name in names if name in blocked|set(case['exclude_from_treatment'])),
                   status_before_anchor=row['status'])
        # Trimming a readout pool cannot justify discarding a previously measured
        # systemic direction. Preserve the unmasked information dimension.
        if row['r_grid'] < reference['r_grid']:
            row['status'] = 'boundary_readout_dimension'
        if row['status'] == 'reported':
            X, C = res['X'], res['Cov']
            result = fit(c['T'], c['Y'], X[:, a], X[:, W], X[:, Z], row['r_grid'], C,
                         300, 20261031+a)
            for target, values in result['targets'].items():
                row.update({target+'_'+key: value for key, value in values.items()})
            row['inference'] = 'fixed boundary and selected design; conditional reference interval'
        records.append(row)
        print(case['study'], case['exposure'], row['status'], 'original dimension', reference['r_grid'], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUT/'structured_cases.csv', index=False)


if __name__ == '__main__':
    import sys
    if '--construction' in sys.argv:
        construction_case()
    else:
        main()
