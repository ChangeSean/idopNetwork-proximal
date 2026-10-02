"""Frozen joint-readout TCGA analysis, separately archived from prior results.

Patient bootstrap fixes roles/rank and refits joint projection and censoring.
Construction resampling reports ordinary bootstrap and 80% subsample stability.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from run_application import load_cohort
from joint_readout_bridge import estimate, select_joint_space
from multiscale_bridge import design_grid
from causal_survival import survival_pseudo_outcomes
from proximal import ols_effect, bh
from diagnose_construction import resample_cohort

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/joint_readout_application_20261001'


def fit(T,D,A,W,Z,r,C,boots,seed,horizon=36):
    B=np.column_stack([A,C]);pseudo=survival_pseudo_outcomes(T,D,horizon,B)
    records={s:dict(estimate=estimate(pseudo[s],A,W,Z,r,C),
                    naive=ols_effect(pseudo[s],A,C)[0]) for s in ('rmst','survival')}
    draws={s:[] for s in records};ndraws={s:[] for s in records}
    rng=np.random.default_rng(seed);failed=0
    for b in range(boots):
        ix=rng.integers(0,len(A),len(A))
        try:
            py=survival_pseudo_outcomes(T[ix],D[ix],horizon,B[ix])
            values={s:estimate(py[s],A[ix],W[ix],Z[ix],r,C[ix]) for s in records}
            if not np.isfinite(list(values.values())).all():raise ValueError('Nonfinite bridge')
            for s in records:
                draws[s].append(values[s]);ndraws[s].append(ols_effect(py[s],A[ix],C[ix])[0])
        except (ValueError,np.linalg.LinAlgError,FloatingPointError):failed+=1
    for s in records:
        se=float(np.std(draws[s],ddof=1)) if len(draws[s])>=max(20,.9*boots) else np.nan
        records[s].update(se_boot=se,ci_low=records[s]['estimate']-norm.ppf(.975)*se,
                          ci_high=records[s]['estimate']+norm.ppf(.975)*se,
                          naive_se_boot=float(np.std(ndraws[s],ddof=1)) if np.isfinite(se) else np.nan,
                          boot_success=len(draws[s]),boot_failed=failed)
    return dict(targets=records,min_g=pseudo['min_g'],median_g=pseudo['median_g'])


def overlap(left,right):
    left=set(filter(None,left.split(';')));right=set(filter(None,right.split(';')))
    return len(left&right)/len(left|right) if left|right else 1.


def run(study,boots=300,construction=60):
    OUT.mkdir(exist_ok=True,parents=True)
    manifest=dict(study=study,boots=boots,construction=construction,
                  algorithm='joint_readout frozen 20261001',
                  source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                                 ['joint_readout_bridge.py','multiscale_bridge.py','probe_overlapping_windows.py']},
                  inference='fixed roles/rank; refit projection, both stages and censoring',
                  resampling='fixed original 60-protein panel and median imputations; rebuild remaining construction')
    (OUT/f'{study}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    cohort=load_cohort(study,'OS',survival=True);grid=design_grid(cohort);reference=grid[0]
    X,C=reference['X'],reference['Cov'];rows=[];start=time.time()
    for a in range(len(reference['names'])):
        row,Z,W,chosen=select_joint_space(grid,a)
        row.update(n=len(X),events=int(cohort['Y'].sum()),horizon=36,components=chosen['ncomp'])
        if row['status']=='reported':
            try:
                fitted=fit(cohort['T'],cohort['Y'],X[:,a],X[:,W],X[:,Z],row['r_bridge'],C,boots,20261003+a)
                row.update(min_g=fitted['min_g'],median_g=fitted['median_g'])
                for s,values in fitted['targets'].items():row.update({s+'_'+k:v for k,v in values.items()})
                if not np.isfinite(row['rmst_se_boot']):row['status']='bootstrap_incomplete'
            except (ValueError,np.linalg.LinAlgError) as error:
                row.update(status='censoring_or_numerical_support',reason=str(error))
        rows.append(row)
        if (a+1)%10==0:print(study,a+1,'clinical exposures',round(time.time()-start,1),flush=True)
    result=pd.DataFrame(rows)
    for s in ('rmst','survival'):
        ok=result.status.eq('reported') & result[s+'_se_boot'].notna()
        result.loc[ok,s+'_p']=2*norm.sf(abs(result.loc[ok,s+'_estimate']/result.loc[ok,s+'_se_boot']))
        result.loc[ok,s+'_q']=bh(result.loc[ok,s+'_p'].to_numpy())
    result.to_csv(OUT/f'{study}_OS_causal_survival.csv',index=False)
    print(study,'clinical fits',result.status.value_counts().to_dict(),flush=True)
    for mode in ('bootstrap','subsample80'):
        rng=np.random.default_rng(20261001 if mode=='bootstrap' else 20261002)
        construction_rows=[]
        for b in range(construction):
            ix=rng.integers(0,len(X),len(X)) if mode=='bootstrap' else rng.choice(len(X),int(.8*len(X)),replace=False)
            sampled,_=resample_cohort(cohort,X,ix);new_grid=design_grid(sampled)
            for a in range(len(rows)):
                new,Z,W,_=select_joint_space(new_grid,a)
                new.update(replicate=b,mode=mode,z_jaccard=overlap(new['Z'],rows[a]['Z']),
                           w_jaccard=overlap(new['W'],rows[a]['W']),
                           exact_roles=new['Z']==rows[a]['Z'] and new['W']==rows[a]['W'],
                           rank_match=new['r_bridge']==rows[a]['r_bridge'])
                construction_rows.append(new)
            if (b+1)%10==0:
                pd.DataFrame(construction_rows).to_csv(OUT/f'{study}_{mode}_construction_replicates.csv',index=False)
                print(study,mode,b+1,round(time.time()-start,1),flush=True)
        cr=pd.DataFrame(construction_rows)
        if not len(cr):continue
        cr.to_csv(OUT/f'{study}_{mode}_construction_replicates.csv',index=False)
        cr['eligible']=cr.status.eq('reported')
        cs=cr.groupby('exposure').agg(report_frequency=('eligible','mean'),
                    exact_role_frequency=('exact_roles','mean'),rank_frequency=('rank_match','mean'),
                    z_jaccard=('z_jaccard','mean'),w_jaccard=('w_jaccard','mean'),replicates=('replicate','size')).reset_index()
        cs['report_mcse']=np.sqrt(cs.report_frequency*(1-cs.report_frequency)/construction)
        cs.to_csv(OUT/f'{study}_{mode}_construction.csv',index=False)
    print(study,'finished',round(time.time()-start,1),'seconds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',choices=['ov','luad'])
    p.add_argument('--boot',type=int,default=300);p.add_argument('--construction',type=int,default=60)
    args=p.parse_args();run(args.study,args.boot,args.construction)
