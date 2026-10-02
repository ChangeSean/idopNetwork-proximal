"""TCGA idop-proximal effects on 36-month RMST and survival probability.

python run_causal_application.py ov --boot 300 --selection-boot 60
python run_causal_application.py luad --boot 300 --selection-boot 60
"""
import argparse
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from proximal import proxy_roles, screen, bridge_information_rank, readout_information_rank, proxy_strength, bh
from run_application import load_cohort, analyse_cohort
from causal_survival import fit_survival_bridge
from idop_core import data_transformation, data_quasi_dynamic

ROOT = Path(__file__).resolve().parent
ALPHAS=(.15,.20,.25,.30)


def analyse_design_grid(cohort,k=5):
    return [analyse_cohort(cohort,alpha=alpha,k=k,rank_mode='bridge',estimate_legacy=False) for alpha in ALPHAS]


def configuration(res, a, nu_min=.05, treatment_eligible=None):
    X, C = res['X'], res['Cov']
    eligible=np.ones(len(res['names']),bool) if treatment_eligible is None else np.asarray(treatment_eligible,bool)
    if eligible.shape!=(len(res['names']),):raise ValueError('Treatment eligibility must follow the protein ordering')
    Z, W = proxy_roles(res['Und'], a, eligible,
                       np.linalg.norm(res['latent']['Lam'], axis=1), 1,limit_to_treatment=False)
    row = dict(exposure=res['names'][a], nZ=len(Z), nW=len(W),
               Z=';'.join(res['names'][j] for j in Z), W=';'.join(res['names'][j] for j in W),
               r_pca=res['latent']['r'], r_bridge=0, nu=np.nan)
    if not screen(Z, W, 1,overcomplete=True):
        row['status'] = 'proxy_dimensions'; return row, Z, W
    info = bridge_information_rank(X[:, W], X[:, Z], X[:, a], C)
    anchor=bridge_information_rank(X[:,W],X[:,[a]+Z],None,C)
    anchor['views']=[W,[a]+Z]
    row.update(r_joint=anchor['r'],r_signal=info['r'],joint_response_dim=len(anchor['views'][0]),joint_predictor_dim=len(anchor['views'][1]))
    row['r_bridge'] = anchor['r']
    row['canonical_roots'] = ';'.join(f'{v:.6g}' for v in info['roots'])
    if not info['r'] or not anchor['r']:
        row['status'] = 'no_shared_direction'; return row, Z, W
    if info['r'] != anchor['r'] or not screen(Z,W,anchor['r'],overcomplete=True):
        row['status'] = 'proxy_rank_coverage';return row,Z,W
    row['nu'] = proxy_strength(X[:, W], X[:, Z], X[:, a], anchor['r'], C)
    row['status'] = 'reported' if row['nu'] >= nu_min else 'weak_bridge_information'
    return row, Z, W


def configuration_grid(resolutions,a,nu_min=.05,treatment_eligible=None):
    """Match a bridge to the information available across a fixed support grid.

    A small separated readout pool can hide a latent direction even when its
    own joint and conditional ranks agree. Estimate the maximum joint rank
    before selecting a design; then take the first sufficiently strong design
    that supplies that dimension conditionally. No outcome enters this choice.
    """
    if not resolutions:raise ValueError('The support grid is empty')
    candidates=[(*configuration(res,a,nu_min,treatment_eligible),res) for res in resolutions]
    dimension=max(row.get('r_joint',0) for row,_,_,_ in candidates)
    selected=next((candidate for candidate in candidates
                   if candidate[0]['status']=='reported' and candidate[0].get('r_joint',0)==dimension),None)
    if selected is None:
        selected=next(candidate for candidate in candidates if candidate[0].get('r_joint',0)==dimension)
    row,Z,W,res=selected
    row.update(r_grid=dimension,alpha=res['alpha'],
               grid_alphas=';'.join(str(r['alpha']) for r in resolutions),
               grid_joint_ranks=';'.join(str(c[0].get('r_joint',0)) for c in candidates),
               grid_conditional_ranks=';'.join(str(c[0].get('r_signal',0)) for c in candidates))
    return row,Z,W,res


def causal_application(study, horizon=36., n_boot=300, selection_boot=60):
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    cohort = load_cohort(study, 'OS', survival=True)
    resolutions=analyse_design_grid(cohort);res=resolutions[0]
    rows = []; started = time.time()
    for a in range(len(res['names'])):
        row, Z, W, chosen = configuration_grid(resolutions, a)
        row.update(n=len(cohort['Y']), events=int(cohort['Y'].sum()), horizon=horizon,
                   windows=chosen['k'], components=chosen['ncomp'])
        if row['status'] == 'reported':
            try:
                fit = fit_survival_bridge(cohort['T'], cohort['Y'], res['X'][:, a], res['X'][:, W],
                                          res['X'][:, Z], row['r_bridge'], res['Cov'], horizon,
                                          n_boot=n_boot, seed=20260930+a)
                row.update(min_g=fit['min_g'], median_g=fit['median_g'])
                for target, values in fit['targets'].items():
                    row.update({target+'_'+key: value for key, value in values.items()})
                if n_boot and not np.isfinite(row['rmst_se_boot']): row['status'] = 'bootstrap_incomplete'
            except (ValueError, np.linalg.LinAlgError) as error:
                row['status'] = 'censoring_or_numerical_support'; row['reason'] = str(error)
        rows.append(row)
        if (a+1) % 10 == 0: print(study, a+1, 'exposures', round(time.time()-started, 1), 'seconds', flush=True)
    output = pd.DataFrame(rows)
    for target in ('rmst', 'survival'):
        if target+'_estimate' not in output: continue
        valid = (output.status == 'reported') & output[target+'_se_boot'].notna()
        output.loc[valid, target+'_t'] = output.loc[valid, target+'_estimate']/output.loc[valid, target+'_se_boot']
        output.loc[valid, target+'_q'] = bh(2*stats.norm.sf(np.abs(output.loc[valid, target+'_t'])))
    result_dir = ROOT/'results'; result_dir.mkdir(exist_ok=True)
    output.to_csv(result_dir/f'{study}_OS_causal_survival.csv', index=False)
    if selection_boot:
        counts = np.zeros(len(rows), int); matches = np.zeros(len(rows), int); rank_matches = np.zeros(len(rows), int)
        z_overlap=np.zeros(len(rows));w_overlap=np.zeros(len(rows));construction_records=[]
        def jaccard(left,right):
            left=set(filter(None,left.split(';')));right=set(filter(None,right.split(';')))
            return len(left&right)/len(left|right) if left|right else 1.
        rng = np.random.default_rng(20261001); n = len(res['X'])
        for b in range(selection_boot):
            ix = rng.integers(0, n, n)
            Xt = data_transformation(pd.DataFrame(res['X'][ix], columns=res['names']), 'Zscore_shift')
            order = np.argsort(Xt.sum(axis=1).to_numpy(), kind='stable')
            coh = dict(qd=data_quasi_dynamic(Xt), Y=cohort['Y'][ix][order], Cov=cohort['Cov'][ix][order],
                       T=cohort['T'][ix][order], names=res['names'])
            br_grid=analyse_design_grid(coh)
            for a in range(len(rows)):
                new, _, _, br = configuration_grid(br_grid, a)
                zj=jaccard(new['Z'],rows[a]['Z']);wj=jaccard(new['W'],rows[a]['W'])
                z_overlap[a]+=zj;w_overlap[a]+=wj
                construction_records.append({**new,'replicate':b,'z_jaccard':zj,'w_jaccard':wj,
                                             'components':br['ncomp']})
                counts[a] += new['status'] == 'reported'
                matches[a] += new['Z'] == rows[a]['Z'] and new['W'] == rows[a]['W']
                rank_matches[a] += new['r_bridge'] == rows[a]['r_bridge']
            if (b+1) % 20 == 0: print(study, 'construction bootstrap', b+1, flush=True)
        stability = pd.DataFrame(dict(exposure=res['names'], report_frequency=counts/selection_boot,
                                     exact_role_frequency=matches/selection_boot,
                                     rank_frequency=rank_matches/selection_boot,
                                     z_jaccard=z_overlap/selection_boot,w_jaccard=w_overlap/selection_boot,
                                     replicates=selection_boot))
        stability['report_mcse']=np.sqrt(stability.report_frequency*(1-stability.report_frequency)/selection_boot)
        stability.to_csv(result_dir/f'{study}_OS_proxy_construction.csv', index=False)
        pd.DataFrame(construction_records).to_csv(result_dir/f'{study}_OS_proxy_construction_replicates.csv',index=False)
    print(study, 'finished', output.status.value_counts().to_dict(), round(time.time()-started, 1), 'seconds', flush=True)
    return output


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('study', choices=['ov','luad'])
    ap.add_argument('--horizon', type=float, default=36.); ap.add_argument('--boot', type=int, default=300)
    ap.add_argument('--selection-boot', type=int, default=60)
    args = ap.parse_args()
    causal_application(args.study, args.horizon, args.boot, args.selection_boot)
