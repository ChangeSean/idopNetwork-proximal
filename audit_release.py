"""Check current-paper claims, reproducible building and frozen analysis sources."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append(name)

    def read(path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    text = (ROOT / 'manuscript.md').read_text(encoding='utf-8')
    main_text = text.split('## Figure legends', 1)[0]
    sections = re.findall(r'^#{2,4} (.+)$', text, re.M)
    check('unique section titles', len(sections) == len(set(sections)))
    paragraphs = [p.strip() for p in main_text.split('\n\n') if len(p.split()) > 35 and not p.startswith('|')]
    check('no duplicate long prose paragraphs', len(paragraphs) == len(set(paragraphs)))
    check('one current workflow, development in supplement', 'earlier 50:50' not in main_text and '### 6.5' not in main_text)
    check('supporting allocation provenance retained', 'overall linear coverage 0.905' in text)
    check('supporting resampling retained', '1/100, 0/100 and 0/100' in text)
    check('component precision tradeoff retained', 'RMSE increases from 0.884 to 1.356' in text)
    check('weak-system bounded fraction retained', '21/200 linear sets' in text)
    check('clinical uncertainty retained', '[-2.07, 4.24]' in text and 'All endpoint sets include zero' in text)
    check('primary ovarian panel complete', '24/60' in main_text and '55 real-line RMST sets' in main_text)
    check('three focused ovarian examples', all(p in main_text for p in ('SERPINE1', 'CCNE1', 'CHEK2')))
    check('ten final clinical cohorts retained', '600 cohort-specific protein contrasts' in text and '58/600' in text)
    check('point directions identified as exploratory', 'exploratory signals' in text and 'All endpoint sets include zero' in text)
    check('post-result example selection identified', 'examples were chosen after analysis' in text)
    before = (ROOT / 'manuscript.md').read_bytes()
    subprocess.run([sys.executable, str(ROOT / 'build_current_manuscript.py')], cwd=ROOT, check=True, capture_output=True)
    check('table refresh preserves canonical manuscript bytes', before == (ROOT / 'manuscript.md').read_bytes())
    builder = (ROOT / 'build_current_manuscript.py').read_text(encoding='utf-8')
    check('builder has no archived manuscript dependency', 'revision_archive' not in builder and 'extend_manuscript' not in builder)
    for directory in ('discovery_estimation_validation_20261001', 'discovery_estimation_application_20261001'):
        for path in (ROOT / 'results' / directory).glob('*manifest*.json'):
            record = json.loads(path.read_text(encoding='utf-8'))
            for name, expected in record.get('source_sha256', {}).items():
                check(f'{path.name}: {name}', hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected)
    pairs = read(ROOT / 'results/joint_readout_validation_20261001/paired_differences.csv')
    check('full paired comparison family present', len(pairs) == 126)
    featured = [('rank1_n1000_modules', 'global_joint_readout'), ('rank1_n1000_modules', 'correlation_joint_readout'),
                ('rank2_n1000_branch', 'conditional_projection_same_design'),
                ('rank2_n352_p60_branch', 'global_joint_readout'), ('rank2_n352_p60_branch', 'correlation_joint_readout')]
    for scenario, comparator in featured:
        row = next(x for x in pairs if x['scenario'] == scenario and x['comparator'] == comparator and x['outcome'] == 'linear')
        check(f'paired estimate in prose: {scenario}/{comparator}', f"{float(row['mse_difference']):.4f}" in text)
        check(f'paired MCSE in prose: {scenario}/{comparator}', f"{float(row['mse_difference_mcse']):.4f}" in text)
    for study in ('ov', 'luad'):
        rows = read(ROOT / f'results/discovery_estimation_application_20261001/{study}_all_exposures.csv')
        check(f'{study} complete panel', len(rows) == 60)
        expected = {'bounded': 2, 'disconnected': 3, 'all_real': 55} if study == 'ov' else {'all_real': 60}
        check(f'{study} confidence-set shapes', Counter(x['rmst_kind'] for x in rows) == expected)
    figure = json.loads((ROOT / 'results/discovery_estimation_application_20261001/figure3_discovery_provenance.json').read_text(encoding='utf-8'))
    designs_path = ROOT / 'results/discovery_estimation_application_20261001/ov_discovery_designs.json'
    designs = json.loads(designs_path.read_text(encoding='utf-8'))
    ov_manifest = json.loads((ROOT / 'results/discovery_estimation_application_20261001/ov_manifest.json').read_text(encoding='utf-8'))
    check('Figure 3 current ovarian cases', [p['exposure'] for p in figure['panels']] == ['PTEN', 'SERPINE1', 'CCNE1'])
    check('Figure 3 discovery sample', figure['discovery_id_sha256'] == ov_manifest['discovery_id_sha256'])
    check('Figure 3 frozen design fingerprint', figure['design_file_sha256'] == hashlib.sha256(designs_path.read_bytes()).hexdigest())
    check('Figure 3 clinical records unchanged', figure['clinical_records_sha256'] == hashlib.sha256((ROOT / 'results/discovery_estimation_application_20261001/ov_all_exposures.csv').read_bytes()).hexdigest())
    check('Figure 3 design replay completed', figure['all_frozen_designs_replayed'] and figure['analysis_csv_files_unchanged'] > 0)
    for panel in figure['panels']:
        protein = panel['exposure']
        row = designs[protein]['row']
        check(f'Figure 3 {protein} patient count', panel['n_discovery'] == ov_manifest['n_discovery'])
        for key in ('Z', 'W', 'r_grid', 'r_signal', 'window_fraction', 'alpha'):
            check(f'Figure 3 {protein} {key}', panel[key] == row[key])
        diagnostics = panel['decomposition_diagnostics']
        check(f'Figure 3 {protein} additive closure', diagnostics['closure_error'] < 1e-9)
        check(f'Figure 3 {protein} centred residuals', abs(diagnostics['mean_residual']) < 1e-9)
    settings = figure['ode_settings']
    for key, expected in dict(smoothing='GCV cubic smoothing spline', basis='shifted Legendre',
                             degree=1, ridge=1., integration='normalised niche time',
                             grid_points=50, selection='fixed molecular support').items():
        check('Figure 3 ODE ' + key, settings[key] == expected)
    for suffix, expected in figure['figures'].items():
        check('Figure 3 ' + suffix + ' fingerprint', hashlib.sha256((ROOT / f'figures/fig3_causal.{suffix}').read_bytes()).hexdigest() == expected)
    for name, expected in figure['reference_figures'].items():
        check(name + ' fingerprint', hashlib.sha256((ROOT / 'figures' / name).read_bytes()).hexdigest() == expected)
    s1, s2 = figure['reference_panels']
    check('S1 full-cohort support retained', s1['n'] == 411 and s1['n_proteins'] == 140 and
          s1['support_edges'] == 261 and s1['components'] == 18)
    reference = ROOT / 'results/joint_readout_application_20261001/figure_network_provenance.csv'
    lck = next(x for x in read(reference) if x['exposure'] == 'LCK')
    check('S2 full-cohort reference roles retained', s2['n'] == 411 and s2['n_proteins'] == 60 and
          s2['Z'] == lck['Z'] and s2['W'] == lck['W'])
    check('Molecular curve settings documented', 'first-degree time-integrated Legendre term' in text and
          'standardised ridge penalty one' in text)
    check('Figure 3 caption matches current panels', 'Panels a,c,e' in text and 'Panels b,d,f' in text)
    result = {'passed': True, 'checks': len(checks), 'canonical_source': 'manuscript.md',
              'manuscript_sha256': hashlib.sha256(before).hexdigest(), 'builder_idempotent': True,
              'final_analysis_sources_unchanged': True, 'complete_comparisons_retained': True}
    (ROOT / 'results/release_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
