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
    result = {'passed': True, 'checks': len(checks), 'canonical_source': 'manuscript.md',
              'manuscript_sha256': hashlib.sha256(before).hexdigest(), 'builder_idempotent': True,
              'final_analysis_sources_unchanged': True, 'complete_comparisons_retained': True}
    (ROOT / 'results/release_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
