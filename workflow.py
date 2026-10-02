"""One command entry point for the published idopNetwork–proximal workflow."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify', 'tables', 'figures', 'word', 'simulate', 'clinical', 'package'])
    parser.add_argument('--cohort', choices=['ov', 'luad', 'both'], default='both')
    args = parser.parse_args()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    if args.stage == 'verify':
        run('check_independent_design.py')
        run('summarize_discovery_estimation.py')
        run('select_method_evidence.py')
        run('audit_release.py')
    elif args.stage == 'tables':
        run('build_current_manuscript.py')
    elif args.stage == 'figures':
        run('make_independent_design_figures.py')
    elif args.stage == 'word':
        run('build_current_manuscript.py')
        run('build_docx.py')
        run('audit_causal_revision.py')
    elif args.stage == 'simulate':
        for scenario in range(7):
            run('validate_discovery_estimation.py', '--scenario', scenario, '--reps', 200, '--boot', 100)
        run('summarize_discovery_estimation.py')
    elif args.stage == 'clinical':
        for cohort in (('ov', 'luad') if args.cohort == 'both' else (args.cohort,)):
            run('run_discovery_estimation_application.py', cohort, '--boot', 300)
        if args.cohort == 'both':
            run('run_independent_structured_cases.py')
            run('audit_discovery_estimation_application.py')
    elif args.stage == 'package':
        run('audit_current_work.py')
        run('package_current_work.py')


if __name__ == '__main__':
    main()
