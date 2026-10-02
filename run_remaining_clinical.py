"""Apply the frozen clinical implementation to the eight remaining TCGA cohorts."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/additional_cohorts_application_20261002'
STUDIES = ('blca', 'brca', 'coadread', 'kirc', 'lgg', 'skcm', 'stad', 'ucec')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(study):
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    import run_discovery_estimation_application as application
    application.OUT = OUT
    with (OUT / f'{study}_run.log').open('w', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                application.run(study, boots=300)
                return {'study': study, 'status': 'completed'}
            except Exception:
                traceback.print_exc()
                return {'study': study, 'status': 'failed', 'error': traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(studies=list(STUDIES), split_seed=20261201, discovery_fraction=.75,
                bootstrap_seed=20261301, bootstrap_draws=300, horizon_months=36,
                exposure_unit='one discovery-cohort SD', panel_size=60,
                implementation='run_discovery_estimation_application.run; unchanged',
                scope='All eight remaining cohorts; no outcome-dependent cohort or split selection',
                source_sha256={name: digest(ROOT / name) for name in (
                    'run_discovery_estimation_application.py', 'independent_design_bridge.py',
                    'conditional_design_bridge.py', 'fieller_bridge.py', 'causal_survival.py',
                    'multiscale_bridge.py', 'DISCOVERY_ESTIMATION_PROTOCOL_20261001.md')},
                input_sha256={study: {kind: digest(ROOT / f'data/tcga_{study}_{kind}.csv')
                                     for kind in ('clinical', 'rppa_long')} for study in STUDIES})
    (OUT / 'extension_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    states = {study: 'pending' for study in STUDIES}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, study): study for study in STUDIES}
        for future in as_completed(futures):
            result = future.result()
            states[result['study']] = result['status']
            (OUT / 'batch_status.json').write_text(json.dumps(states, indent=2), encoding='utf-8')
            print(json.dumps(result), flush=True)
    if any(value != 'completed' for value in states.values()):
        raise SystemExit('Some cohorts failed; inspect cohort logs.')
    print('Completed all eight cohorts with the frozen implementation.', flush=True)


if __name__ == '__main__':
    main()
