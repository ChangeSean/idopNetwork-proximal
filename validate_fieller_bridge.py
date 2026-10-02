"""Fresh paired Fieller and normal confidence evaluation for concentrated bridges."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from validate_conditional_design import SETTINGS, extended_system, sample_system, cohort_from
from evaluate_causal_method import truth_graph, role_validity
from multiscale_bridge import design_grid
from conditional_design_bridge import select_conditional_design
from causal_survival import survival_pseudo_outcomes
import fieller_bridge as FB

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/concentrated_bridge_validation_20261001'


def run(scenario, reps=200, boots=100):
    OUT.mkdir(parents=True, exist_ok=True)
    setting = SETTINGS[scenario]
    system = extended_system(setting)
    graph, factors, ynode = truth_graph(system)
    a = system['a0']
    manifest = dict(scenario=scenario, reps=reps, boots=boots, offset=100000,
                    molecular_selection='conditional_design', inference='construction-conditioned; joint P and G refit',
                    source_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                                   ('fieller_bridge.py', 'conditional_design_bridge.py', 'validate_fieller_bridge.py')})
    (OUT/f'manifest_part{scenario}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    records = []
    start = time.time()
    for rep in range(reps):
        lin = sample_system(system, setting, 100000+rep, 'linear')
        sur = sample_system(system, setting, 100000+rep, 'rmst')
        raw = lin['X']
        c, order = cohort_from(raw)
        grid = design_grid(c)
        row, Z, W, chosen = select_conditional_design(grid, a)
        X, A = grid[0]['X'], raw[order, a]
        rng = np.random.default_rng(20360000+scenario*10000+rep)
        draws = [rng.integers(0, len(raw), len(raw)) for _ in range(boots)]
        responses = {'linear': lin['Y'][order],
                     'rmst': survival_pseudo_outcomes(sur['T'][order], sur['D'][order], 36, censor='km')['rmst']}
        boot_sur = [survival_pseudo_outcomes(sur['T'][order][ix], sur['D'][order][ix], 36, censor='km')['rmst'] for ix in draws]
        for target, y in responses.items():
            truth = lin['target'] if target == 'linear' else sur['target']
            entry = dict(row, scenario=setting['name'], replicate=rep, outcome=target, target=truth)
            if Z and W:
                entry.update(role_validity(graph, factors, ynode, a, Z, W))
            if row['status'] == 'reported':
                try:
                    result = FB.fit(y, A, X[:, W], X[:, Z], row['r_grid'], draws=draws,
                                    responses=boot_sur if target == 'rmst' else None)
                    N, D = result['N'], result['D']
                    V = np.asarray(result['covariance'])
                    _, derivative = FB.point_and_gradient(np.r_[N, D])
                    se = np.sqrt(max(float(derivative@V@derivative), 0.))
                    entry.update(estimate=result['estimate'], error=result['estimate']-truth,
                                 N=json.dumps(N), D=json.dumps(D), contrasts=result['contrasts'],
                                 set_kind=result['kind'], set_intervals=json.dumps(result['intervals']),
                                 fieller_covered=FB.contains(result, truth),
                                 normal_covered=abs(result['estimate']-truth) <= norm.ppf(.975)*se,
                                 normal_width=2*norm.ppf(.975)*se,
                                 fieller_width=result['intervals'][0][1]-result['intervals'][0][0] if result['kind'] == 'bounded' else np.nan,
                                 nuisance_sv=result['pivots']['nuisance_sv'], boot_success=result['boot_success'],
                                 pivots=json.dumps(result['pivots']), rank_correct=row['r_grid'] == system['r'])
                except (ValueError, np.linalg.LinAlgError) as error:
                    entry.update(status='nuisance_support', reason=str(error))
            records.append(entry)
        if (rep+1) % 20 == 0 or rep+1 == reps:
            pd.DataFrame(records).to_csv(OUT/f'replicates_part{scenario}.csv', index=False)
            print(setting['name'], rep+1, '/', reps, 'seconds', round(time.time()-start, 1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', type=int, required=True)
    p.add_argument('--reps', type=int, default=200)
    p.add_argument('--boot', type=int, default=100)
    args = p.parse_args()
    run(args.scenario, args.reps, args.boot)
