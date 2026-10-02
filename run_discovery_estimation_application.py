"""Clinical independent discovery/estimation with discovery-only preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from multiscale_bridge import design_grid
from idop_core import data_quasi_dynamic
from causal_survival import survival_pseudo_outcomes
import independent_design_bridge as ID

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/discovery_estimation_application_20261001'


def prepare(study):
    rp = pd.read_csv(ROOT/f'data/tcga_{study}_rppa_long.csv')
    cl = pd.read_csv(ROOT/f'data/tcga_{study}_clinical.csv', index_col=0)
    wide = rp.pivot_table(index='patientId', columns='gene', values='value')
    ids = sorted(set(cl.index) & set(wide.index))
    cl = cl.loc[ids]
    cl = cl[cl.OS_STATUS.notna() & pd.to_numeric(cl.OS_MONTHS, errors='coerce').notna()]
    wide = wide.loc[cl.index]
    allocation = np.random.default_rng(20261201).permutation(len(wide))
    d, e = allocation[:3*len(wide)//4], allocation[3*len(wide)//4:]
    train = wide.iloc[d]
    train = train.loc[:, train.notna().mean() >= .9]
    med = train.median()
    train = train.fillna(med)
    corr = train.corr().to_numpy().copy()
    np.fill_diagonal(corr, 0)
    drop = set()
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            if corr[i, j] > .999 and train.columns[i] not in drop:
                drop.add(train.columns[j])
    train = train.drop(columns=sorted(drop))
    names = list(train.var().sort_values(ascending=False).index[:60])
    mean, sd = train[names].mean(), train[names].std(ddof=0)
    if (sd <= 0).any():
        raise ValueError('Discovery panel has constant proteins')
    xd = (train[names]-mean)/sd
    xe = ((wide.iloc[e][names].fillna(med[names])-mean)/sd).to_numpy()
    shifted = xd-xd.min()+1
    order = np.argsort(shifted.sum(1).to_numpy(), kind='stable')
    covariates = []
    age = pd.to_numeric(cl.AGE, errors='coerce')
    if age.iloc[d].notna().mean() > .5:
        covariates.append(age.fillna(age.iloc[d].median()).to_numpy()[:, None])
    for cat in ('SUBTYPE', 'SEX'):
        if cat in cl and cl.iloc[d][cat].notna().mean() > .5 and cl.iloc[d][cat].nunique() > 1:
            mode = cl.iloc[d][cat].mode().iloc[0]
            filled = cl[cat].fillna(mode)
            levels = sorted(cl.iloc[d][cat].dropna().unique())
            covariates.append(np.column_stack([filled.eq(v).astype(float).to_numpy() for v in levels[1:]]))
    C = np.column_stack(covariates) if covariates else np.zeros((len(cl), 0))
    events = cl.OS_STATUS.str.startswith('1').astype(float).to_numpy()
    times = pd.to_numeric(cl.OS_MONTHS, errors='coerce').to_numpy()
    cohort = dict(qd=data_quasi_dynamic(shifted), names=names, Y=np.zeros(len(d)),
                  Cov=C[d][order], T=None)
    info = dict(study=study, n=len(wide), n_discovery=len(d), n_estimation=len(e),
                discovery_events=int(events[d].sum()), estimation_events=int(events[e].sum()),
                names=names, discovery_mean=mean.to_dict(), discovery_sd=sd.to_dict(),
                split_seed=20261201,
                # Digests prove the stored split without publishing patient identifiers.
                discovery_id_sha256=hashlib.sha256('\n'.join(wide.index[d]).encode()).hexdigest(),
                estimation_id_sha256=hashlib.sha256('\n'.join(wide.index[e]).encode()).hexdigest())
    return cohort, xe, C[e], times[e], events[e], info


def run(study, boots=300):
    OUT.mkdir(parents=True, exist_ok=True)
    cohort, X, C, T, D, info = prepare(study)
    grid = design_grid(cohort)
    rng = np.random.default_rng(20261301)
    draws = [rng.integers(0, len(X), len(X)) for _ in range(boots)]
    rows, designs = [], {}
    for a, name in enumerate(cohort['names']):
        design = ID.freeze(grid, a)
        designs[name] = design
        row = dict(design['row'], study=study, n_discovery=info['n_discovery'],
                   n_estimation=len(X), estimation_events=int(D.sum()),
                   exposure_sd=info['discovery_sd'][name], design_ready=design['ready'])
        results = {}
        if design['ready']:
            try:
                py = survival_pseudo_outcomes(T, D, 36, np.column_stack([X[:, a], C]))
                boot, failed_censor = [], 0
                for ix in draws:
                    try:
                        boot.append(survival_pseudo_outcomes(T[ix], D[ix], 36, np.column_stack([X[ix, a], C[ix]])))
                    except (ValueError, np.linalg.LinAlgError):
                        failed_censor += 1
                        boot.append({target: np.full(len(ix), np.nan) for target in ('rmst', 'survival')})
                row['censor_boot_failures'] = failed_censor
                row['min_g'] = py['min_g']
                for target in ('rmst', 'survival'):
                    results[target] = ID.fit(py[target], X[:, a], X[:, design['W']], X[:, design['Z']], C,
                                             design, draws, [v[target] for v in boot])
            except (ValueError, np.linalg.LinAlgError) as error:
                results = {target: ID.unavailable('censoring_support') for target in ('rmst', 'survival')}
                row['reason'] = str(error)
        else:
            results = {target: ID.unavailable('discovery_'+design['row']['status']) for target in ('rmst', 'survival')}
        for target, result in results.items():
            for key in ('estimate', 'kind', 'status', 'boot_success', 'boot_attempts'):
                row[target+'_'+key] = result[key]
            row[target+'_intervals'] = json.dumps(result['intervals'])
        rows.append(row)
        print(study, name, 'design', design['row']['status'], 'inference', row['rmst_status'], flush=True)
    pd.DataFrame(rows).to_csv(OUT/f'{study}_all_exposures.csv', index=False)
    (OUT/f'{study}_discovery_designs.json').write_text(json.dumps(designs, indent=2), encoding='utf-8')
    files = ('independent_design_bridge.py', 'run_discovery_estimation_application.py',
             'DISCOVERY_ESTIMATION_PROTOCOL_20261001.md', 'conditional_design_bridge.py', 'fieller_bridge.py')
    info.update(boots=boots, horizon=36, exposure_unit='one discovery-cohort SD',
                inference='single independent 75:25 split; all discovery decisions fixed',
                source_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (OUT/f'{study}_manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(study, 'finished', pd.DataFrame(rows).rmst_status.value_counts().to_dict(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('study', choices=['ov', 'luad'])
    p.add_argument('--boot', type=int, default=300)
    args = p.parse_args()
    run(args.study, args.boot)
