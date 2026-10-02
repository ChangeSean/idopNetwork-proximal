"""Frozen-rule validation of joint-space proximal estimation on new seeds."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from evaluate_causal_method import SCENARIOS,sample_system,truth_graph,role_validity
from evaluate_window_prototype import extended_system
from idop_core import data_transformation,data_quasi_dynamic
from run_causal_application import configuration_grid
from multiscale_bridge import design_grid
from joint_readout_bridge import select_joint_space,estimate
from proximal import proximal_bridge,ols_effect,latent_state
from causal_survival import survival_pseudo_outcomes

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'joint_readout_validation_20261001'
SETTINGS=SCENARIOS+[
    dict(name='rank1_n411_p60_branch',r=1,n=411,topology='branch',latent=.25,seed=11,n_o=47),
    dict(name='rank2_n352_p60_branch',r=2,n=352,topology='branch',latent=.35,seed=7,n_o=47)]


def effect(method,y,A,X,F,spec):
    if method=='oracle_factor':return ols_effect(y,A,F)[0]
    if method=='unadjusted':return ols_effect(y,A)[0]
    if method=='factor_adjustment':return ols_effect(y,A,latent_state(X)['F'])[0]
    row,Z,W=spec
    if method=='joint_readout':return estimate(y,A,X[:,W],X[:,Z],row['r_bridge'])
    return proximal_bridge(y,A,X[:,W],X[:,Z],row['r_bridge'],min_sv_ratio=1e-8)[0]


def summaries(table):
    rows=[]
    for (scenario,outcome,method),data in table.groupby(['scenario','outcome','method']):
        fits=data[data.status.eq('reported')];n=len(fits)
        errors=fits.error.to_numpy();covered=np.asarray(fits.covered,dtype=float)
        rows.append(dict(scenario=scenario,outcome=outcome,method=method,attempts=len(data),
                         reported=n,report_rate=n/len(data),bias=errors.mean() if n else np.nan,
                         bias_mcse=errors.std(ddof=1)/np.sqrt(n) if n>1 else np.nan,
                         rmse=np.sqrt(np.mean(errors**2)) if n else np.nan,
                         coverage=covered.mean() if n else np.nan,
                         coverage_mcse=np.sqrt(covered.mean()*(1-covered.mean())/n) if n else np.nan,
                         width=fits.width.mean(),rank_correct=fits.rank_correct.mean(),
                         valid_roles=fits.set_valid.mean(),median_nu=fits.nu.median()))
    return pd.DataFrame(rows)


def run(scenario,reps=200,boots=100,offset=30000):
    OUT.mkdir(exist_ok=True,parents=True);setting=SETTINGS[scenario]
    manifest=dict(scenario=scenario,setting=setting,reps=reps,boots=boots,offset=offset,
                  source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                                 ['joint_readout_bridge.py','multiscale_bridge.py','probe_overlapping_windows.py']},
                  parameters={'widths':[.2,.4],'penalties':[.15,.20,.25,.30],
                              'rank_level':'min(.05,n**(-.5))','selection':'maximum retained joint root',
                              'projection':'first-stage W on (1,A,Z), centered on (1) before SVD',
                              'conditional_completeness':'identification assumption; measured conditional rank is reported',
                              'strength':'reported quantitatively; no fixed strength cutoff'},
                  bootstrap='fixed roles/rank; projection and censoring re-estimated',
                  independence='new patient seeds after rule freeze; does not validate full construction intervals')
    (OUT/f'manifest_part{scenario}.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    system=extended_system(setting);a=system['a0'];graph,factors,y_node=truth_graph(system)
    records=[];start=time.time()
    for rep in range(reps):
        linear=sample_system(system,setting,offset+rep,'linear')
        survival=sample_system(system,setting,offset+rep,'rmst')
        assert np.array_equal(linear['X'],survival['X'])
        raw=linear['X'];names=[f'v{j}' for j in range(system['p'])]
        scaled=data_transformation(pd.DataFrame(raw,columns=names),'Zscore_shift')
        order=np.argsort(scaled.sum(1).to_numpy(),kind='stable')
        cohort=dict(qd=data_quasi_dynamic(scaled),names=names,Y=linear['Y'][order],Cov=np.zeros((setting['n'],0)),T=None)
        grid=design_grid(cohort);X=grid[0]['X'];A=raw[order,a];F=linear['F'][order]
        row,Z,W,_=select_joint_space(grid,a)
        old,Z0,W0,_=configuration_grid(grid[:4],a)
        methods={'joint_readout':(row,Z,W),'conditional_projection_same_design':(row,Z,W),
                 'prior_grid':(old,Z0,W0),'oracle_factor':({},[],[]),
                 'unadjusted':({},[],[]),'factor_adjustment':({},[],[])}
        rng=np.random.default_rng(10200000+10000*setting['seed']+rep+offset)
        draws=[rng.integers(0,setting['n'],setting['n']) for _ in range(boots)]
        T,D=survival['T'][order],survival['D'][order]
        responses={'linear':linear['Y'][order],'rmst':survival_pseudo_outcomes(T,D,36,censor='km')['rmst']}
        boot_rmst=[survival_pseudo_outcomes(T[ix],D[ix],36,censor='km')['rmst'] for ix in draws]
        for outcome,y in responses.items():
            target=linear['target'] if outcome=='linear' else survival['target']
            for method,spec in methods.items():
                rr,zz,ww=spec;status=rr.get('status','reported')
                entry={**rr,'scenario':setting['name'],'replicate':rep,'outcome':outcome,'method':method,
                       'status':status,'target':target,'true_r':system['r'],'n':setting['n'],'p':system['p'],
                       'boot_requested':boots}
                if zz and ww:entry.update(role_validity(graph,factors,y_node,a,zz,ww))
                if status=='reported':
                    try:
                        e=effect(method,y,A,X,F,spec);values=[]
                        for b,ix in enumerate(draws):
                            yy=y[ix] if outcome=='linear' else boot_rmst[b]
                            try:
                                value=effect(method,yy,A[ix],X[ix],F[ix],spec)
                                if np.isfinite(value):values.append(value)
                            except (ValueError,np.linalg.LinAlgError):pass
                        se=float(np.std(values,ddof=1)) if len(values)>1 else np.nan
                        entry.update(estimate=e,error=e-target,se=se,boot_success=len(values),
                                     covered=float(abs(e-target)<=norm.ppf(.975)*se),width=2*norm.ppf(.975)*se)
                        if 'r_bridge' in rr:entry['rank_correct']=float(rr['r_bridge']==system['r'])
                        if len(values)<max(20,.9*boots):entry['status']='bootstrap_incomplete'
                    except (ValueError,np.linalg.LinAlgError) as error:entry.update(status='numerical_support',reason=str(error))
                records.append(entry)
        if (rep+1)%20==0:
            pd.DataFrame(records).to_csv(OUT/f'replicates_part{scenario}.csv',index=False)
            print(setting['name'],rep+1,'elapsed',round(time.time()-start,1),flush=True)
    table=pd.DataFrame(records);table.to_csv(OUT/f'replicates_part{scenario}.csv',index=False)
    result=summaries(table);result.to_csv(OUT/f'summary_part{scenario}.csv',index=False)
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scenario',type=int,required=True)
    parser.add_argument('--reps',type=int,default=200);parser.add_argument('--boot',type=int,default=100)
    parser.add_argument('--offset',type=int,default=30000);args=parser.parse_args()
    run(args.scenario,args.reps,args.boot,args.offset)
