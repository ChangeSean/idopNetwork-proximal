"""Refresh tables in the canonical manuscript; preserve authored prose.

No archived manuscript or incremental revision script is used. Repeated runs
produce identical Markdown and never rerun or select experiments.
"""
from pathlib import Path
import re
import pandas as pd
from paper_tables import SCENARIOS, ROMAN, metric, table, renumber_references
from clinical_reporting import case_table_rows, cohort_table_rows

ROOT = Path(__file__).resolve().parent


def main():
    text = (ROOT / 'manuscript.md').read_text(encoding='utf-8')
    representation = pd.read_csv(ROOT / 'results/joint_readout_validation_20261001/all_summary.csv')
    final = pd.read_csv(ROOT / 'results/discovery_estimation_validation_20261001/all_summary.csv')

    def replace(number, headers, rows):
        nonlocal text
        pattern = rf'^\*\*Table {number}\.\*\*[^\n]*\n\n(?:\|[^\n]*\n)+'
        found = re.search(pattern, text, re.M)
        assert found, number
        caption = found.group().split('\n', 1)[0]
        text = text[:found.start()] + table(caption, headers, rows) + text[found.end():]

    rows = []
    for outcome in ('linear', 'rmst'):
        for scenario in SCENARIOS:
            r = representation[(representation.outcome == outcome) & (representation.scenario == scenario) & (representation.method == 'joint_readout')].iloc[0]
            rows.append([outcome.upper(), ROMAN[scenario], f'{int(r.reported)}/{int(r.attempts)}',
                         metric(r.bias, r.bias_mcse), metric(r.rmse, r.rmse_mcse), metric(r.coverage, r.coverage_mcse), f'{100*r.rank_correct:.1f}'])
    replace(2, ['Outcome', 'System', 'Reported', 'Bias', 'RMSE', 'Coverage', 'Correct rank (%)'], rows)
    methods = [('joint_readout', 'Joint-quality readout'), ('conditional_projection_same_design', 'Same design, conditional projection'),
               ('prior_grid', 'Previous grid bridge'), ('global_joint_readout', 'One global window'),
               ('correlation_joint_readout', 'Correlation proxies'), ('designated_joint_readout', 'Designated joint readout'),
               ('designated_proximal', 'Designated proximal'), ('unadjusted', 'Unadjusted'),
               ('factor_adjustment', 'Factor adjustment'), ('oracle_factor', 'Oracle factors')]
    rows = []
    for scenario in (SCENARIOS[2], SCENARIOS[3], SCENARIOS[6]):
        for method, label in methods:
            r = representation[(representation.outcome == 'linear') & (representation.scenario == scenario) & (representation.method == method)].iloc[0]
            values = [f'{r[k]:.3f}' if r.reported >= 5 else '—' for k in ('bias', 'rmse', 'coverage')]
            rows.append([ROMAN[scenario], label, f'{int(r.reported)}/{int(r.attempts)}'] + values)
    replace(3, ['System', 'Estimator', 'Reported', 'Bias', 'RMSE', 'Coverage'], rows)
    rows = []
    for outcome in ('linear', 'rmst'):
        for scenario in SCENARIOS:
            r = final[(final.outcome == outcome) & (final.scenario == scenario)].iloc[0]
            rows.append([outcome.upper(), ROMAN[scenario], f'{int(r.point_fits)}/{int(r.attempts)}', f'{r.rank_correct:.3f}',
                         metric(r.bias, r.bias_mcse), metric(r.rmse, r.rmse_mcse), metric(r.coverage, r.coverage_mcse), f'{int(r.bounded)}/{int(r.attempts)}'])
    replace(4, ['Target', 'System', 'Point fits', 'Correct rank', 'Bias (MCSE)', 'RMSE (MCSE)', 'Set coverage', 'Bounded sets'], rows)
    replace(5, ['Protein', 'Treatment proxies', 'Rank', 'RMST months and set', 'Survival points and set'], case_table_rows())
    replace('S1', ['Cohort', 'Discovery / estimation', 'Estimation deaths', 'Designs', 'Point fits',
                   'RMST B/D/R', 'Survival B/D/R'], cohort_table_rows())
    text = renumber_references(re.sub(r'\n{3,}', '\n\n', text))
    (ROOT / 'manuscript.md').write_text(text, encoding='utf-8', newline='\n')
    section = text.split('## 4 Identification and estimation', 1)[1].split('## 5 Simulation', 1)[0]
    appendix = text.split('## Appendix A', 1)[1].split('## Supplementary tables', 1)[0]
    if (ROOT / 'revised_theory.md').exists():
        (ROOT / 'revised_theory.md').write_text('# Identification and independent bridge inference\n\n' + section + '\n## Appendix A' + appendix, encoding='utf-8')
    print('Refreshed Tables 2–5 and S1 from aggregate results; canonical prose preserved.')


if __name__ == '__main__':
    main()
