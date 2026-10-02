"""Fresh validation of conditional design and construction-aware inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from validate_joint_readout_bridge import SETTINGS
from evaluate_window_prototype import extended_system
from evaluate_causal_method import sample_system, truth_graph, role_validity
from idop_core import data_transformation, data_quasi_dynamic
from multiscale_bridge import design_grid
from conditional_design_bridge import select_conditional_design, bootstrap_interval
from joint_readout_bridge import select_joint_space, estimate
from causal_survival import survival_pseudo_outcomes

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/conditional_design_validation_20261001'


def cohort_from(raw):
    names = [f'v{j}' for j in range(raw.shape[1])]
    scaled = data_transformation(pd.DataFrame(raw, columns=names), 'Zscore_shift')
    order = np.argsort(scaled.sum(1).to_numpy(), kind='stable')
    return dict(qd=data_quasi_dynamic(scaled), names=names, Y=np.zeros(len(raw)),
                Cov=np.zeros((len(raw), 0)), T=None), order


def run(scenario, reps=200, boots=100, full=False, offset=50000):
    OUT.mkdir(parents=True, exist_ok=True)
    setting = SETTINGS[scenario]
    system = extended_system(setting)
    a = system['a0']
    graph, factors, ynode = truth_graph(system)
    mode = 'full' if full else 'fixed'
    manifest = dict(scenario=scenario, setting=setting, reps=reps, boots=boots, offset=offset,
                    mode=mode, seed_boot=20261100,
                    selection='full conditional rank; maximum smallest conditional root',
                    exposure_units='original raw generating exposure unit in every draw',
                    source_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                                   ('conditional_design_bridge.py', 'validate_conditional_design.py',
                                    'joint_readout_bridge.py', 'multiscale_bridge.py')})
    (OUT/f'{mode}_manifest_part{scenario}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows, construction = [], []
    start = time.time()
    for rep in range(reps):
        lin = sample_system(system, setting, offset+rep, 'linear')
        surv = sample_system(system, setting, offset+rep, 'rmst')
        assert np.array_equal(lin['X'], surv['X'])
        raw = lin['X']
        cohort, order = cohort_from(raw)
        grid = design_grid(cohort)
        X = grid[0]['X']
        selection = select_conditional_design(grid, a)
        previous = select_joint_space(grid, a)
        specs = {'conditional_design': selection}
        if full:
            specs['conditional_fixed_reference'] = selection
        else:
            specs['joint_quality_design'] = previous
        responses = dict(linear=lin['Y'][order],
                         rmst=survival_pseudo_outcomes(surv['T'][order], surv['D'][order], 36, censor='km')['rmst'])
        points, draws = {}, {}
        for method, (row, Z, W, selected) in specs.items():
            draws[method] = {'linear': [], 'rmst': []}
            for outcome, response in responses.items():
                entry = dict(row, scenario=setting['name'], replicate=rep, outcome=outcome, method=method,
                             inference=mode, attempts_boot=boots, true_rank=system['r'],
                             target=lin['target'] if outcome == 'linear' else surv['target'])
                if Z and W:
                    entry.update(role_validity(graph, factors, ynode, a, Z, W))
                if row['status'] == 'reported':
                    entry['estimate'] = estimate(response, raw[order, a], X[:, W], X[:, Z], row['r_bridge'])
                    entry['error'] = entry['estimate']-entry['target']
                    entry['rank_correct'] = row['r_bridge'] == system['r']
                points[method, outcome] = entry
        if any(v['status'] == 'reported' for v in points.values()):
            rng = np.random.default_rng(20261100+scenario*10000+rep+offset)
            for b in range(boots):
                ix = rng.integers(0, len(raw), len(raw))
                if full:
                    # Resample original unsorted patient records; reorder every target.
                    sampled, bo = cohort_from(raw[ix])
                    bg = design_grid(sampled)
                    row, Z, W, _ = select_conditional_design(bg, a)
                    bx, ba = bg[0]['X'], raw[ix][bo, a]
                    yy = lin['Y'][ix][bo]
                    tb, db = surv['T'][ix][bo], surv['D'][ix][bo]
                    active = {'conditional_design': (row, Z, W, {})}
                    construction.append(dict(row, scenario=setting['name'], replicate=rep, bootstrap=b))
                else:
                    bx, ba = X[ix], raw[order, a][ix]
                    yy = responses['linear'][ix]
                    tb, db = surv['T'][order][ix], surv['D'][order][ix]
                    active = specs
                py = survival_pseudo_outcomes(tb, db, 36, censor='km')['rmst']
                if full and selection[0]['status'] == 'reported':
                    rr, zz, ww, _ = selection
                    fixed_x = X[np.argsort(order)][ix]
                    fixed_y = lin['Y'][ix]
                    fixed_pseudo = survival_pseudo_outcomes(surv['T'][ix], surv['D'][ix], 36, censor='km')['rmst']
                    for outcome, y in [('linear', fixed_y), ('rmst', fixed_pseudo)]:
                        val = estimate(y, raw[ix, a], fixed_x[:, ww], fixed_x[:, zz], rr['r_bridge'])
                        if np.isfinite(val):
                            draws['conditional_fixed_reference'][outcome].append(val)
                for method, (row, Z, W, _) in active.items():
                    if row['status'] != 'reported' or points[method, 'linear']['status'] != 'reported':
                        continue
                    try:
                        for outcome, y in [('linear', yy), ('rmst', py)]:
                            val = estimate(y, ba, bx[:, W], bx[:, Z], row['r_bridge'])
                            if np.isfinite(val):
                                draws[method][outcome].append(val)
                    except (ValueError, np.linalg.LinAlgError):
                        pass
        for (method, outcome), entry in points.items():
            if entry['status'] == 'reported':
                ci = bootstrap_interval(entry['estimate'], draws[method][outcome], boots)
                entry.update(ci)
                entry['status'] = 'reported' if np.isfinite(ci['se_boot']) else 'construction_incomplete'
                entry['covered'] = ci['ci_low'] <= entry['target'] <= ci['ci_high'] if entry['status'] == 'reported' else np.nan
                entry['width'] = ci['ci_high']-ci['ci_low']
            rows.append(entry)
        if (rep+1) % 10 == 0 or rep+1 == reps:
            pd.DataFrame(rows).to_csv(OUT/f'{mode}_replicates_part{scenario}.csv', index=False)
            if full:
                pd.DataFrame(construction).to_csv(OUT/f'{mode}_construction_part{scenario}.csv', index=False)
            print(mode, setting['name'], rep+1, '/', reps, 'seconds', round(time.time()-start, 1), flush=True)
    print(mode, setting['name'], 'finished', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', type=int, required=True)
    p.add_argument('--reps', type=int, default=200)
    p.add_argument('--boot', type=int, default=100)
    p.add_argument('--full', action='store_true')
    p.add_argument('--offset', type=int, default=50000)
    args = p.parse_args()
    run(args.scenario, args.reps, args.boot, args.full, args.offset)
