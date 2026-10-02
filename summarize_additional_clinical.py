"""Audit the eight-cohort extension and summarise all ten final clinical analyses."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from run_discovery_estimation_application import prepare
from run_remaining_clinical import STUDIES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/additional_cohorts_application_20261002'
ORIGINAL = ROOT / 'results/discovery_estimation_application_20261001'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def isolation(study):
    original = prepare(study)
    real_read = pd.read_csv
    rp = real_read(ROOT / f'data/tcga_{study}_rppa_long.csv')
    cl = real_read(ROOT / f'data/tcga_{study}_clinical.csv', index_col=0)
    wide = rp.pivot_table(index='patientId', columns='gene', values='value')
    included = cl.loc[sorted(set(cl.index) & set(wide.index))]
    included = included[included.OS_STATUS.notna() & pd.to_numeric(included.OS_MONTHS, errors='coerce').notna()]
    order = np.random.default_rng(20261201).permutation(len(included))
    held_out = included.index[order[3 * len(included) // 4:]]
    altered = rp.copy()
    mask = altered.patientId.isin(held_out)
    altered.loc[mask, 'value'] = 1000 + altered.loc[mask, 'value'] * 9
    changed_cl = cl.copy()
    changed_cl.loc[held_out, 'OS_MONTHS'] = 1.
    changed_cl.loc[held_out, 'OS_STATUS'] = '0:LIVING'

    def fake_read(path, *args, **kwargs):
        if Path(path).name == f'tcga_{study}_rppa_long.csv':
            return altered.copy()
        if Path(path).name == f'tcga_{study}_clinical.csv':
            return changed_cl.copy()
        return real_read(path, *args, **kwargs)

    with patch('pandas.read_csv', side_effect=fake_read):
        changed = prepare(study)
    assert original[0]['qd'].equals(changed[0]['qd'])
    assert np.array_equal(original[0]['Cov'], changed[0]['Cov'])
    assert original[0]['names'] == changed[0]['names']
    for field in ('discovery_mean', 'discovery_sd', 'discovery_id_sha256', 'estimation_id_sha256'):
        assert original[-1][field] == changed[-1][field]
    assert not np.array_equal(original[1], changed[1])
    assert not np.array_equal(original[3], changed[3])
    return original[-1]


def main():
    plan = json.loads((OUT / 'extension_plan.json').read_text(encoding='utf-8'))
    checks = 0
    for name, expected in plan['source_sha256'].items():
        assert digest(ROOT / name) == expected, name
        checks += 1
    for study, inputs in plan['input_sha256'].items():
        for kind, expected in inputs.items():
            assert digest(ROOT / f'data/tcga_{study}_{kind}.csv') == expected
            checks += 1
    summaries, bounded, reasons = [], [], []
    for study in (*STUDIES, 'ov', 'luad'):
        folder = OUT if study in STUDIES else ORIGINAL
        meta = json.loads((folder / f'{study}_manifest.json').read_text(encoding='utf-8'))
        data = pd.read_csv(folder / f'{study}_all_exposures.csv')
        designs = json.loads((folder / f'{study}_discovery_designs.json').read_text(encoding='utf-8'))
        assert len(data) == 60 and not data.exposure.duplicated().any()
        assert set(data.exposure) == set(meta['names']) == set(designs)
        assert meta['boots'] == 300 and meta['horizon'] == 36 and meta['split_seed'] == 20261201
        assert meta['n_discovery'] == 3 * meta['n'] // 4
        checks += 4
        for name, expected in meta['source_sha256'].items():
            assert digest(ROOT / name) == expected
            checks += 1
        if study in STUDIES:
            prepared = isolation(study)
            for field in ('n', 'n_discovery', 'n_estimation', 'estimation_events',
                          'names', 'discovery_id_sha256', 'estimation_id_sha256'):
                assert prepared[field] == meta[field]
            checks += 16
        for row in data.to_dict('records'):
            design = designs[row['exposure']]
            assert bool(row['design_ready']) == design['ready']
            checks += 1
            for target in ('rmst', 'survival'):
                intervals = json.loads(row[target + '_intervals'])
                status = row[target + '_status']
                kind = row[target + '_kind']
                if status == 'estimated':
                    assert design['ready'] and row[target + '_boot_success'] >= 293
                    assert row[target + '_boot_attempts'] == 300
                    assert np.isfinite(row[target + '_estimate'])
                else:
                    assert intervals == [[-np.inf, np.inf]] and pd.isna(row[target + '_estimate'])
                assert all(lo <= hi for lo, hi in intervals)
                if kind == 'bounded':
                    assert len(intervals) == 1 and np.isfinite(intervals).all()
                    bounded.append(dict(study=study, exposure=row['exposure'], target=target,
                                        estimate=row[target + '_estimate'], low=intervals[0][0],
                                        high=intervals[0][1], contains_zero=intervals[0][0] <= 0 <= intervals[0][1]))
                checks += 3
        for target in ('rmst', 'survival'):
            shapes = data[target + '_kind'].value_counts().to_dict()
            nonzero, directional = 0, 0
            for text in data[target + '_intervals']:
                intervals = json.loads(text)
                nonzero += bool(intervals) and not any(lo <= 0 <= hi for lo, hi in intervals)
                directional += bool(intervals) and (all(lo > 0 for lo, _ in intervals)
                                                     or all(hi < 0 for _, hi in intervals))
            summaries.append(dict(study=study, target=target, n=meta['n'],
                                  n_discovery=meta['n_discovery'], n_estimation=meta['n_estimation'],
                                  estimation_events=meta['estimation_events'], exposures=len(data),
                                  discovery_ready=int(data.design_ready.sum()),
                                  point_fits=int(data[target + '_estimate'].notna().sum()),
                                  **{kind: shapes.get(kind, 0) for kind in
                                     ('bounded', 'disconnected', 'all_real', 'empty', 'half_line')},
                                  nonempty_sets_excluding_zero=int(nonzero),
                                  directional_sets=int(directional)))
            reasons.extend(dict(study=study, target=target, status=status, count=int(count))
                           for status, count in data[target + '_status'].value_counts().items())
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'all_ten_cohorts_summary.csv', index=False)
    pd.DataFrame(bounded).to_csv(OUT / 'bounded_effect_sets.csv', index=False)
    pd.DataFrame(reasons).to_csv(OUT / 'inference_status_counts.csv', index=False)
    audit = dict(passed=True, checks=checks, additional_cohorts=8, additional_exposures=480,
                 additional_endpoint_sets=960, all_final_cohorts=10, all_final_exposures=600,
                 source_and_input_hashes='match extension plan',
                 original_implementation='unchanged',
                 discovery_isolation='Held-out protein values and survival labels perturbed in all eight added cohorts; discovery construction unchanged',
                 patient_identifiers_in_outputs=False,
                 confidence_level='95% per exposure and endpoint; no cross-protein or cross-cohort multiplicity adjustment',
                 output_sha256={name: digest(OUT / name) for name in
                                ('all_ten_cohorts_summary.csv', 'bounded_effect_sets.csv', 'inference_status_counts.csv')})
    (OUT / 'extension_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    table = ['|队列|总人数|估计集人数|估计集死亡数|可用代理设计|点估计|有界RMST集合|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary[summary.target.eq('rmst')].to_dict('records'):
        table.append(f"|{row['study'].upper()}|{row['n']}|{row['n_estimation']}|{row['estimation_events']}|{row['discovery_ready']}|{row['point_fits']}|{row['bounded']}|")
    report = '\n'.join([
        '# 十个TCGA队列的固定流程结果', '',
        '新增八个队列全部完成，每队列分析60个蛋白、两个36个月生存目标、300次估计集重抽样。',
        '原有OV/LUAD结果未重算，固定实现及输入哈希核对通过。', '', *table, '',
        '新增八队列的960个置信集合全部为全实数；其中32个蛋白完成点估计，102个可用设计在删失或重抽样环节未完成推断。',
        '其余346个蛋白在发现阶段没有得到可用设计。', '',
        '十队列共600个蛋白：RMST有2个有界集合、3个不连通集合、595个全实数集合；',
        '36个月生存概率有2个有界集合、1个不连通集合、597个全实数集合。',
        '全部非空集合包含零，没有方向明确的效应集合。', '',
        'OV的PTEN：RMST估计+1.019个月，95%集合[-2.066,4.237]个月；',
        '36个月生存概率估计+6.473个百分点，95%集合[-6.770,19.307]个百分点。',
        'OV的HSPA1A：RMST估计-0.624个月，95%集合[-21.973,10.188]个月。', '',
        '新增队列中的失败位置：COADREAD和STAD的全部可用设计未通过36个月删失支持检查；',
        'BRCA、LGG、UCEC主要受重抽样完成率限制。KIRC的14个可用设计全部完成点估计，',
        '但置信集合仍然全部为全实数，表明完成计算与获得效应信息是两个不同环节。', '',
        '当前结论：现有数据与固定实现尚未产生明确的医学干预效应发现；',
        '这些结果不等于蛋白没有作用，也不单独判定方法在其他数据条件下的表现。', '',
        '详细结果：各队列*_all_exposures.csv；集合形状汇总all_ten_cohorts_summary.csv；',
        '失败原因inference_status_counts.csv；核对记录extension_audit.json。', ''
    ])
    (OUT / 'RESULTS.md').write_text(report, encoding='utf-8')
    print(json.dumps(audit, indent=2))
    print(summary[summary.target.eq('rmst')].to_string(index=False))


if __name__ == '__main__':
    main()
