"""Audit every split-study attempt and summarize explicit inference denominators."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import fieller_bridge as FB

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/discovery_estimation_validation_20261001'


def se_probability(p, n):
    return np.sqrt(p*(1-p)/n) if n else np.nan


def main():
    parts, checks = [], 0
    for s in range(7):
        m = json.loads((OUT/f'manifest_part{s}.json').read_text())
        d = pd.read_csv(OUT/f'replicates_part{s}.csv')
        assert len(d) == 2*m['reps'] and m['offset'] == 140000
        assert not d.duplicated(['replicate', 'outcome']).any()
        for f, sha in m['source_sha256'].items():
            assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest() == sha, f
            checks += 1
        for row in d.itertuples():
            result = dict(intervals=json.loads(row.set_intervals))
            assert FB.contains(result, row.target) == row.covered
            assert row.n_discovery+row.n_estimation == m['setting']['n']
            if row.status != 'estimated':
                assert row.set_kind == 'all_real' and np.isnan(row.estimate)
            else:
                assert row.boot_success >= .975*m['boots'] and row.boot_success >= 20
            checks += 3
        parts.append(d)
    data = pd.concat(parts, ignore_index=True)
    summaries = []
    for (scenario, outcome), group in data.groupby(['scenario', 'outcome'], sort=False):
        fits = group[group.status.eq('estimated')]
        errors = fits.error.to_numpy()
        n, nfit = len(group), len(fits)
        bias = errors.mean() if nfit else np.nan
        rmse = np.sqrt(np.mean(errors**2)) if nfit else np.nan
        coverage = group.covered.mean()
        fit_cov = fits.covered.mean()
        normal = fits.normal_covered.mean()
        rank = fits.rank_correct.mean()
        row = dict(scenario=scenario, outcome=outcome, attempts=n,
                   discovery_ready=int(group.design_ready.sum()), point_fits=nfit,
                   point_rate=nfit/n, point_rate_mcse=se_probability(nfit/n, n),
                   valid_roles=fits.set_valid.mean(), rank_correct=rank,
                   rank_correct_mcse=se_probability(rank, nfit),
                   bias=bias, bias_mcse=np.std(errors, ddof=1)/np.sqrt(nfit) if nfit>1 else np.nan,
                   rmse=rmse, rmse_mcse=np.std(errors**2, ddof=1)/(2*rmse*np.sqrt(nfit)) if nfit>1 and rmse>0 else np.nan,
                   coverage=coverage, coverage_mcse=se_probability(coverage, n),
                   fit_coverage=fit_cov, fit_coverage_mcse=se_probability(fit_cov, nfit),
                   normal_coverage=normal, normal_coverage_mcse=se_probability(normal, nfit),
                   bounded_width=fits.loc[fits.set_kind.eq('bounded'), 'bounded_width'].mean(),
                   fallback=int((group.status != 'estimated').sum()))
        for shape in ('bounded', 'disconnected', 'all_real', 'half_line', 'empty'):
            row[shape] = int(group.set_kind.eq(shape).sum())
        assert sum(row[k] for k in ('bounded', 'disconnected', 'all_real', 'half_line', 'empty')) == n
        summaries.append(row)
    data.to_csv(OUT/'all_replicates.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'all_summary.csv', index=False)
    (OUT/'validation_audit.json').write_text(json.dumps(dict(passed=True, checks=checks,
                   attempts=len(data), datasets=len(data)//2, protocol_offset=140000), indent=2))
    print(summary[['scenario','outcome','point_fits','rank_correct','bias','rmse','coverage','fit_coverage','bounded','all_real']].to_string(index=False))
    print('passed', checks, 'checks')


if __name__ == '__main__':
    main()
