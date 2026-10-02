"""Shared clinical presentation from every stored final exposure record."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'results/discovery_estimation_application_20261001'
EXTENSION = ROOT / 'results/additional_cohorts_application_20261002'
STUDIES = ('blca', 'brca', 'coadread', 'kirc', 'lgg', 'luad', 'ov', 'skcm', 'stad', 'ucec')
# Worked examples chosen after results: two OV bounded cases, the same PTEN
# exposure in KIRC, and a completed LUAD contrast within both target ranges.
CASES = (('ov', 'PTEN'), ('kirc', 'PTEN'), ('ov', 'HSPA1A'), ('luad', 'TGM2'))


def folder(study):
    return BASE if study in ('ov', 'luad') else EXTENSION


def records():
    frames = []
    for study in STUDIES:
        frame = pd.read_csv(folder(study) / f'{study}_all_exposures.csv')
        assert len(frame) == 60 and frame.exposure.is_unique
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def point_records(data=None):
    data = records() if data is None else data
    points = data[data.rmst_status.eq('estimated') & data.survival_status.eq('estimated')].copy()
    assert points[['rmst_estimate', 'survival_estimate']].notna().all().all()
    points['rmst_outside_target_range'] = points.rmst_estimate.abs() > 36
    points['survival_outside_target_range'] = points.survival_estimate.abs() > 1
    columns = ['study', 'exposure', 'n_discovery', 'n_estimation', 'estimation_events',
               'r_grid', 'Z', 'W', 'rmst_estimate', 'rmst_kind', 'rmst_intervals',
               'survival_estimate', 'survival_kind', 'survival_intervals',
               'rmst_boot_success', 'survival_boot_success',
               'rmst_outside_target_range', 'survival_outside_target_range']
    return points[columns].sort_values(['study', 'exposure']).reset_index(drop=True)


def format_set(row, target, scale=1):
    estimate = row[target + '_estimate'] * scale
    intervals = json.loads(row[target + '_intervals'])
    if row[target + '_kind'] == 'all_real':
        text = 'All real values'
    else:
        text = ' union '.join(f'[{lo * scale:.2f}, {hi * scale:.2f}]' for lo, hi in intervals)
    return f'{estimate:.2f}; {text}'


def case_records(data=None):
    data = records() if data is None else data
    return [data[data.study.eq(study) & data.exposure.eq(exposure)].iloc[0]
            for study, exposure in CASES]


def case_table_rows():
    return [[row.study.upper(), row.exposure, int(row.r_grid), format_set(row, 'rmst'),
             format_set(row, 'survival', 100)] for row in case_records()]


def cohort_table_rows():
    data = records()
    rows = []
    for study in STUDIES:
        panel = data[data.study.eq(study)]
        shapes = []
        for target in ('rmst', 'survival'):
            counts = panel[target + '_kind'].value_counts()
            shapes.append('/'.join(str(counts.get(kind, 0)) for kind in ('bounded', 'disconnected', 'all_real')))
        rows.append([study.upper(), f'{int(panel.n_discovery.iloc[0])}/{int(panel.n_estimation.iloc[0])}',
                     int(panel.estimation_events.iloc[0]), f'{int(panel.design_ready.sum())}/60',
                     f'{int(panel.rmst_estimate.notna().sum())}/60', *shapes])
    return rows


def main():
    points = point_records()
    points.to_csv(EXTENSION / 'point_estimates.csv', index=False)
    print(f'Reported every completed point estimate: {len(points)} of 600 cohort-specific exposures.')


if __name__ == '__main__':
    main()
