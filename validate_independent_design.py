"""End-to-end validation on fresh datasets with one independent patient split."""
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
from causal_survival import survival_pseudo_outcomes
import independent_design_bridge as ID
import fieller_bridge as FB

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/independent_design_validation_20261001'


def run(scenario, reps=200, boots=100):
    OUT.mkdir(parents=True, exist_ok=True)
    setting = SETTINGS[scenario]
    system = extended_system(setting)
    graph, factors, ynode = truth_graph(system)
    a = system['a0']
    files = ('independent_design_bridge.py', 'validate_independent_design.py',
             'conditional_design_bridge.py', 'joint_readout_bridge.py', 'fieller_bridge.py',
             'multiscale_bridge.py', 'INDEPENDENT_DESIGN_PROTOCOL_20261001.md')
    manifest = dict(scenario=scenario, setting=setting, reps=reps, boots=boots,
                    offset=120000, split_fraction=.5, split_seed=20261200,
                    source_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (OUT/f'manifest_part{scenario}.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows, start = [], time.time()
    for rep in range(reps):
        lin = sample_system(system, setting, 120000+rep, 'linear')
        sur = sample_system(system, setting, 120000+rep, 'rmst')
        assert np.array_equal(lin['X'], sur['X'])
        raw = lin['X']
        discovery, analysis = ID.partition(len(raw), 20261200+10000*scenario+rep)
        cohort, _ = cohort_from(raw[discovery])
        design = ID.freeze(design_grid(cohort), a)
        # Any fixed nonzero rescaling of a selected proxy preserves its bridge.
        # Use raw measurements in estimation; P was learned in standardised coordinates.
        sd = raw[discovery].std(0, ddof=0)
        X = (raw[analysis]-raw[discovery].mean(0))/sd
        A = raw[analysis, a]  # target is per raw generating exposure unit
        Z, W = design['Z'], design['W']
        rng = np.random.default_rng(20261300+10000*scenario+rep)
        draws = [rng.integers(0, len(analysis), len(analysis)) for _ in range(boots)]
        targets = {'linear': lin['Y'][analysis],
                   'rmst': survival_pseudo_outcomes(sur['T'][analysis], sur['D'][analysis], 36, censor='km')['rmst']}
        boot_sur = [survival_pseudo_outcomes(sur['T'][analysis][ix], sur['D'][analysis][ix], 36, censor='km')['rmst']
                    for ix in draws] if design['ready'] else None
        for outcome, y in targets.items():
            truth = lin['target'] if outcome == 'linear' else sur['target']
            result = ID.fit(y, A, X[:, W], X[:, Z], None, design, draws,
                            boot_sur if outcome == 'rmst' else None)
            row = dict(design['row'], scenario=setting['name'], replicate=rep, outcome=outcome,
                       n_discovery=len(discovery), n_estimation=len(analysis), design_ready=design['ready'],
                       true_rank=system['r'], rank_correct=design['row']['r_grid'] == system['r'],
                       target=truth, status=result['status'], set_kind=result['kind'],
                       set_intervals=json.dumps(result['intervals']), estimate=result['estimate'],
                       covered=FB.contains(result, truth), boot_success=result['boot_success'],
                       boot_attempts=result['boot_attempts'])
            if Z and W:
                row.update(role_validity(graph, factors, ynode, a, Z, W))
            if result['status'] == 'estimated':
                row.update(error=result['estimate']-truth,
                           normal_covered=abs(result['estimate']-truth) <= norm.ppf(.975)*result['normal_se'],
                           normal_width=2*norm.ppf(.975)*result['normal_se'],
                           contrasts=result['contrasts'],
                           bounded_width=result['intervals'][0][1]-result['intervals'][0][0]
                           if result['kind'] == 'bounded' else np.nan)
            rows.append(row)
        if (rep+1) % 20 == 0 or rep+1 == reps:
            pd.DataFrame(rows).to_csv(OUT/f'replicates_part{scenario}.csv', index=False)
            print(setting['name'], rep+1, '/', reps, 'seconds', round(time.time()-start, 1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', type=int, required=True)
    p.add_argument('--reps', type=int, default=200)
    p.add_argument('--boot', type=int, default=100)
    args = p.parse_args()
    run(args.scenario, args.reps, args.boot)
