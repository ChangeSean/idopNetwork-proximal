"""Build a traceable editorial evidence selection from existing, unchanged results."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'method_evidence_selection_20261001'
REP = ROOT / 'results' / 'joint_readout_validation_20261001'
FINAL = ROOT / 'results' / 'discovery_estimation_validation_20261001'
CLIN = ROOT / 'results' / 'discovery_estimation_application_20261001'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    rep = read(REP / 'all_summary.csv')
    raw = read(REP / 'all_replicates.csv')
    pairs = read(REP / 'paired_differences.csv')
    final = read(FINAL / 'all_summary.csv')
    clinical = {study: read(CLIN / f'{study}_all_exposures.csv') for study in ('ov', 'luad')}
    checks = 0
    index = {(x['scenario'], x['outcome'], x['method']): x for x in rep}
    raw_index = {}
    for x in raw:
        key = (x['scenario'], x['outcome'], x['method'])
        raw_index.setdefault(key, []).append(x)
    assert len(rep) == 140 and len(final) == 14
    checks += 1
    for key, summary in index.items():
        group = raw_index[key]
        fits = [x for x in group if x['status'] == 'reported']
        assert len(group) == int(summary['attempts']) == 200
        assert len(fits) == int(summary['reported'])
        if fits:
            errors = [float(x['error']) for x in fits]
            assert math.isclose(statistics.mean(errors), float(summary['bias']), abs_tol=1e-10)
            assert math.isclose(math.sqrt(statistics.mean(e * e for e in errors)), float(summary['rmse']), abs_tol=1e-10)
            assert math.isclose(statistics.mean(float(x['covered']) for x in fits), float(summary['coverage']), abs_tol=1e-10)
        checks += 1
    for pair in pairs:
        base = (pair['scenario'], pair['outcome'])
        ours = {x['replicate']: float(x['error']) for x in raw_index[base + ('joint_readout',)] if x['status'] == 'reported'}
        other = {x['replicate']: float(x['error']) for x in raw_index[base + (pair['comparator'],)] if x['status'] == 'reported'}
        common = ours.keys() & other.keys()
        delta = [ours[k] ** 2 - other[k] ** 2 for k in sorted(common)]
        assert len(delta) == int(pair['paired'])
        if delta:
            assert math.isclose(statistics.mean(delta), float(pair['mse_difference']), abs_tol=1e-10)
        if len(delta) > 1:
            assert math.isclose(statistics.stdev(delta) / math.sqrt(len(delta)), float(pair['mse_difference_mcse']), abs_tol=1e-10)
        checks += 1

    complete = []
    for pair in pairs:
        ours = index[(pair['scenario'], pair['outcome'], 'joint_readout')]
        other = index[(pair['scenario'], pair['outcome'], pair['comparator'])]
        complete.append({**pair, 'ours_reported': ours['reported'], 'other_reported': other['reported'],
                         'attempts_each': ours['attempts'], 'ours_rmse': ours['rmse'], 'other_rmse': other['rmse'],
                         'ours_coverage': ours['coverage'], 'other_coverage': other['coverage'],
                         'interpretation': 'paired MSE difference = ours minus comparator; errors and coverage use reported fits'})
    write_csv(OUT / 'all_paired_comparisons.csv', complete)
    selected_keys = {
        ('rank1_n1000_modules', 'linear', 'global_joint_readout'),
        ('rank1_n1000_modules', 'linear', 'correlation_joint_readout'),
        ('rank2_n1000_branch', 'linear', 'conditional_projection_same_design'),
        ('rank2_n352_p60_branch', 'linear', 'global_joint_readout'),
        ('rank2_n352_p60_branch', 'linear', 'correlation_joint_readout'),
    }
    selected = [x for x in complete if (x['scenario'], x['outcome'], x['comparator']) in selected_keys]
    assert len(selected) == 5
    write_csv(OUT / 'featured_comparisons.csv', selected)
    write_csv(OUT / 'all_representation_results.csv', rep)
    write_csv(OUT / 'all_final_validation_results.csv', final)
    overview = []
    for outcome in ('linear', 'rmst'):
        for comparator in sorted({x['comparator'] for x in complete}):
            group = [x for x in complete if x['outcome'] == outcome and x['comparator'] == comparator]
            overview.append({'outcome': outcome, 'comparator': comparator, 'systems': len(group),
                             'paired_mse_lower': sum(float(x['mse_difference']) < 0 for x in group),
                             'paired_mse_higher': sum(float(x['mse_difference']) > 0 for x in group),
                             'note': 'directional counts only; not a significance test or pooled estimate'})
    write_csv(OUT / 'comparison_overview.csv', overview)

    selected_clinical = []
    for x in clinical['ov']:
        if x['exposure'] in ('PTEN', 'HSPA1A'):
            intervals = json.loads(x['rmst_intervals'])
            assert len(intervals) == 1 and intervals[0][0] < 0 < intervals[0][1]
            selected_clinical.append(x)
            checks += 1
    assert len(selected_clinical) == 2
    assert len(clinical['ov']) == len(clinical['luad']) == 60
    assert Counter(x['rmst_kind'] for x in clinical['ov']) == {'bounded': 2, 'disconnected': 3, 'all_real': 55}
    assert Counter(x['rmst_kind'] for x in clinical['luad']) == {'all_real': 60}
    checks += 1
    write_csv(OUT / 'featured_clinical_cases.csv', selected_clinical)

    table = ['| 系统与比较对象 | 我们/对照的报告数 | RMSE：我们/对照 | 配对MSE差（MCSE） | 覆盖率：我们/对照 |',
             '| --- | --- | --- | --- | --- |']
    labels = {'global_joint_readout': '单一全局窗口', 'correlation_joint_readout': '相关性代理',
              'conditional_projection_same_design': '相同设计的条件投影'}
    systems = {'rank1_n1000_modules': 'III：模块化，r=1', 'rank2_n1000_branch': 'IV：弱双因子',
               'rank2_n352_p60_branch': 'VII：352人、60蛋白、r=2'}
    for x in selected:
        table.append(f"| {systems[x['scenario']]} / {labels[x['comparator']]} | {x['ours_reported']}/200；{x['other_reported']}/200 | "
                     f"{float(x['ours_rmse']):.3f} / {float(x['other_rmse']):.3f} | {float(x['mse_difference']):.4f} ({float(x['mse_difference_mcse']):.4f}) | "
                     f"{float(x['ours_coverage']):.3f} / {float(x['other_coverage']):.3f} |")
    memo = '''# idop + proximal：方法主张与重点证据

这份材料从已有完整结果中选择最能说明方法机制的案例，作为后续写作的重点证据。案例选择发生在结果产生之后，属于编辑性归纳，不是预先指定的主要终点。原始结果、模拟设置、患者划分和主稿均未改动。

## 1. 主结论：分子表示和代理构造能改善指定系统中的效应估计

最适合前置的是三个系统：模块化单因子、弱双因子和接近临床规模的60蛋白双因子系统。每个系统有200次尝试，直接比较相同患者和同一干预目标。下面的重点比较同时列出配对误差与报告分母：

''' + '\n'.join(table) + '''

配对MSE差是“我们的平方误差减去对照平方误差”，只用双方均报告的同一重复；负值表示我们的误差较小。表中各自RMSE和覆盖率则使用各自报告样本，分母不同时不把其差值当成配对估计。MCSE衡量模拟误差，没有进行事后多重比较检验。

**可用于论文的主张：** 在模块化系统III中，多窗口网络构造改善了单一全局窗口和相关性代理的效应估计；在系统IV中，联合读出保留了对识别重要的系统方向，相比同一设计的条件投影降低了偏差；在60蛋白系统VII中，网络设计也优于这里实现的两种简单代理方案。

系统IV的线性偏差为0.013，而相同设计条件投影为0.203；覆盖率为0.965和0.625。这是联合读出机制最直接的模拟印证。其RMST比较同时存在精度代价：RMSE为1.356和0.884、覆盖率为0.965和0.900。应据此描述偏差、覆盖和精度的不同变化。

完整矩阵保留全部七个系统、两个结局和十种估计器。在系统IV，单一全局窗口与我们的线性RMSE相近；系统VI中其RMSE为0.109，我们为0.112。系统VII中，已知正确代理角色的proximal参考RMSE为0.093，我们为0.121。方法优势据此定位为特定结构下的代理构造和表示价值，不扩展为所有系统或所有proximal实现的普遍优势。条件信息选择的配对研究也未显示普遍的RMSE改善。

## 2. 最终流程：独立发现与估计有单独的新数据验证

上述表示研究使用原始joint-quality规则，offset=30000，并固定角色/维度计算重抽样区间。最终方法采用条件信息选择、75:25独立分样本和集中矩反演，offset=140000，七个系统各200组新数据、两个结局、各100次重抽样。两项研究回答不同问题，不合并成同一个最终估计器的结果。

最终流程覆盖率为0.950–0.995。信息较充分的系统II、III、V，线性有限区间分别为190/200、187/200、186/200，RMST为190/200、188/200、189/200。这些系统可以同时说明覆盖与实际有限信息。

弱系统IV有136/200个点估计，线性总体覆盖率0.965、点估计可用样本中的覆盖率0.949；单个有限区间21/200，全实数集合163/200，其中64个源于计算/设计不可用。RMST总体覆盖率0.995，单个有限区间同样为21/200。该系统说明矩反演如何表达有限识别信息，不以高覆盖率代替估计精度。

## 3. 临床主案例：PTEN提供最清楚的效应尺度

PTEN适合作为临床主展示案例，是因为它在当前独立流程中有有限区间和清楚的结局单位；其正向点估计不是案例有效性条件。OV共411人，发现308人、估计103人，估计样本67个死亡。

- PTEN每增加一个发现样本标准差，36个月RMST点估计增加1.02个月，95%集合为[-2.07, 4.24]个月；36个月生存概率点估计增加6.47个百分点，集合为[-6.77, 19.31]个百分点。
- HSPA1A作为第二个有限区间案例完整保留：RMST为-0.62个月，集合[-21.97, 10.19]个月，精度较低。
- OV全部60个蛋白中有26个发现设计、24个点估计，RMST集合为2个有限、3个不连通、55个全实数。LUAD全部60个RMST集合均为全实数。主案例与完整队列统计一起呈现。

**可用于论文的主张：** PTEN展示了如何把网络候选代理落实为一个按标准差定义、以生存月份和百分点表达的临床目标，以及当前独立数据所提供的精度。该结果尚未确立PTEN效应的正负方向。

PTEN设计使用CHEK2作处理代理，因果解释依赖该代理的排除条件和桥模型；网络与信息秩不单独证明这一生物学条件。PTEN临床文献也存在不同方向：[2014年研究](https://pubmed.ncbi.nlm.nih.gov/25608477/)发现PTEN降低与较差生存相关，[2020年多中心研究](https://pubmed.ncbi.nlm.nih.gov/32555365/)发现高级别浆液性卵巢癌中胞质PTEN下调与较长生存相关。测量方式与预后关联目标也不同于本研究的分子干预目标，因此两项研究均作为背景，而不将单一方向当成因果效应验证。

## 4. 可直接用于正文的英文结果与讨论文字

The representation study identifies concrete settings in which molecular construction improves effect estimation. In the rank-one modular system, the joint-readout estimator had linear-effect RMSE 0.049, compared with 0.195 for the global-window design and 0.184 for correlation-selected proxies. Paired mean squared error differences were -0.0356 (MCSE 0.0038; 191 paired fits) and -0.0313 (0.0026; 200 paired fits). In the weak two-factor system, joint readout reduced linear bias from 0.203 to 0.013 relative to conditional projection of the same selected design, with coverage increasing from 0.625 to 0.965. In the 60-protein two-factor system, paired mean squared error differences relative to the global-window and correlation designs were -0.0802 (0.0096; 173 pairs) and -0.2544 (0.0072; 200 pairs). These comparisons support the value of network-based proxy construction and joint readout in the specified systems; the full comparison matrix reports both outcomes, every comparator and all reporting denominators.

The final independent workflow was evaluated separately in seven systems with 200 fresh datasets each. Coverage was 0.950–0.995 across the two targets. In systems II, III and V, 186–190 of 200 linear-effect sets and 188–190 RMST sets were bounded. In weak system IV, only 21 of 200 sets were bounded for each target, showing the limited precision alongside coverage. The ovarian PTEN analysis illustrates the clinical scale: a one-standard-deviation contrast corresponded to an estimated 1.02-month difference in 36-month RMST, with a 95% set of [-2.07, 4.24] months. This clinical case connects molecular design to an interpretable survival target and its estimation uncertainty under the specified proximal model.

## 5. 来源及复核

- 表示比较：results/joint_readout_validation_20261001/all_summary.csv、all_replicates.csv、paired_differences.csv。
- 最终流程：results/discovery_estimation_validation_20261001/all_summary.csv。
- 临床：results/discovery_estimation_application_20261001/ov_all_exposures.csv、luad_all_exposures.csv。
- 当前目录保存全部126项配对比较、完整140行表示结果、14行最终结果、5项重点比较及2项临床案例。
- select_method_evidence.py从重复级结果重新核对表示研究的分母、偏差、RMSE、覆盖率和所有配对MSE及MCSE。evidence_audit.json记录来源SHA256和检查数。
'''
    (ROOT / 'METHOD_EVIDENCE_SELECTION_20261001.md').write_text(memo, encoding='utf-8')
    sources = [REP / 'all_summary.csv', REP / 'all_replicates.csv', REP / 'paired_differences.csv',
               FINAL / 'all_summary.csv', CLIN / 'ov_all_exposures.csv', CLIN / 'luad_all_exposures.csv']
    audit = {'passed': True, 'checks': checks, 'representation_rows': len(rep), 'paired_rows': len(complete),
             'final_rows': len(final), 'featured_comparisons': len(selected), 'clinical_cases': len(selected_clinical),
             'selection': 'post-result editorial selection; full comparison matrix retained',
             'method_versions': {'representation_offset': 30000, 'final_offset': 140000, 'final_split': '75:25'},
             'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (OUT / 'evidence_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'source_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
