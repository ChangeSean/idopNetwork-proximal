"""Check point-focused reporting against all 600 final clinical records."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from clinical_reporting import STUDIES, CASES, records, point_records, case_records, EXTENSION

ROOT = Path(__file__).resolve().parent


def main():
    data = records()
    points = point_records(data)
    stored = pd.read_csv(EXTENSION / 'point_estimates.csv')
    checks = 0
    assert len(data) == 600 and len(points) == len(stored) == 58
    assert set(zip(points.study, points.exposure)) == set(zip(stored.study, stored.exposure))
    checks += 2
    for row in stored.to_dict('records'):
        reference = data[data.study.eq(row['study']) & data.exposure.eq(row['exposure'])].iloc[0]
        for target in ('rmst', 'survival'):
            assert reference[target + '_status'] == 'estimated'
            assert np.isclose(row[target + '_estimate'], reference[target + '_estimate'], rtol=0, atol=1e-12)
            assert row[target + '_kind'] == reference[target + '_kind']
            assert json.loads(row[target + '_intervals']) == json.loads(reference[target + '_intervals'])
            checks += 4
    summary = pd.read_csv(EXTENSION / 'all_ten_cohorts_summary.csv')
    for study in STUDIES:
        panel = data[data.study.eq(study)]
        for target in ('rmst', 'survival'):
            row = summary[summary.study.eq(study) & summary.target.eq(target)].iloc[0]
            assert row.exposures == len(panel)
            assert row.point_fits == panel[target + '_estimate'].notna().sum()
            for kind in ('bounded', 'disconnected', 'all_real', 'empty', 'half_line'):
                assert row[kind] == panel[target + '_kind'].eq(kind).sum()
            checks += 7
    text = (ROOT / 'manuscript.md').read_text(encoding='utf-8')
    assert '600 cohort-specific protein contrasts' in text and '58/600' in text
    assert 'All endpoint sets include zero' in text and 'exploratory signals' in text
    assert '379 percentage points' in text and '-53.66 months' in text
    checks += 3
    provenance = json.loads((EXTENSION / 'clinical_figures_provenance.json').read_text(encoding='utf-8'))
    assert provenance['point_estimates_per_panel'] == len(points)
    assert provenance['rmst_off_range'] == points.rmst_outside_target_range.sum()
    assert provenance['survival_off_range'] == points.survival_outside_target_range.sum()
    assert len(case_records(data)) == len(CASES) == 3
    checks += 4
    main_text = text.split('## Figure legends', 1)[0]
    ovarian = points[points.study.eq('ov')]
    assert len(ovarian) == provenance['main_point_estimates_per_panel'] == 24
    assert provenance['main_cohort'] == 'ov' and provenance['main_attempted_exposures'] == 60
    assert provenance['main_rmst_off_range'] == ovarian.rmst_outside_target_range.sum()
    assert provenance['main_survival_off_range'] == ovarian.survival_outside_target_range.sum()
    assert '24/60' in main_text and '55 real-line RMST sets' in main_text
    assert CASES == (('ov', 'PTEN'), ('ov', 'SERPINE1'), ('ov', 'CCNE1'))
    checks += 6
    result = dict(passed=True, checks=checks, cohorts=10, attempted_exposures=600,
                  primary_cohort='ov', main_displayed_point_estimates=24,
                  supplementary_displayed_point_estimates=58, all_point_estimates_retained=True,
                  all_confidence_sets_retained=True, missing_estimates_imputed=False,
                  exploratory_interpretation=True,
                  source_sha256=hashlib.sha256(text.encode()).hexdigest())
    (EXTENSION / 'reporting_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
