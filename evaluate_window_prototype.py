"""Paired, fresh-seed pilot for the fixed overlapping-window prototype.

Thirty replicates are a development comparison, not final coverage evidence.
Clinical outcomes never enter this prototype choice. Existing primary results
are not overwritten. Both estimators share patients and bootstrap indices.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from dgp import make_system
from evaluate_causal_method import SCENARIOS, system_for, sample_system, truth_graph, role_validity
from idop_core import data_transformation, data_quasi_dynamic
from proximal import proximal_bridge
from run_causal_application import analyse_design_grid, configuration_grid
from probe_overlapping_windows import overlapping_grid
from diagnose_construction import OUT


def extended_system(setting):
    if 'n_o' not in setting:
        return system_for(setting)
    system=make_system(r=setting['r'],seed=setting['seed'],n_o=setting['n_o'])
    rng=np.random.default_rng(setting['seed']+500)
    O=system['iO']; theta=system['Theta']; theta[np.ix_(O,O)]=0
    for j in range(1,len(O)):
        parent=(j-1)//2
        theta[O[parent],O[j]]=rng.choice([-1,1])*rng.uniform(.3,.7)
    system['B']=np.linalg.inv(np.eye(system['p'])-theta)
    return system


def run(reps=30, boots=100):
    OUT.mkdir(exist_ok=True,parents=True)
    settings=SCENARIOS+[
        dict(name='rank1_n411_p60_branch',r=1,n=411,topology='branch',latent=.25,seed=11,n_o=47),
        dict(name='rank2_n352_p60_branch',r=2,n=352,topology='branch',latent=.35,seed=7,n_o=47)]
    manifest=dict(settings=settings,reps=reps,boots=boots,replicate_offset=12000,
                  bootstrap_seed_offset=8100000,prototype_window_fraction=.4,
                  fixed_parameters='5 windows, support frequency >.6, alpha grid .15 .20 .25 .30, nu .05',
                  status='development pilot; not promoted to the primary method')
    (OUT/'window_prototype_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    records=[];start=time.time()
    for setting in settings:
        system=extended_system(setting);a=system['a0'];graph,factors,y_node=truth_graph(system)
        for rep in range(reps):
            data=sample_system(system,setting,rep+12000,'linear')
            raw=data['X']; names=[f'v{j}' for j in range(system['p'])]
            scaled=data_transformation(pd.DataFrame(raw,columns=names),'Zscore_shift')
            order=np.argsort(scaled.sum(1).to_numpy(),kind='stable')
            cohort=dict(qd=data_quasi_dynamic(scaled),names=names,Y=data['Y'][order],
                        Cov=np.zeros((setting['n'],0)),T=None)
            grid=analyse_design_grid(cohort)
            if setting['name']==SCENARIOS[0]['name'] and rep==0:
                assert all(np.array_equal(g['A'],h['A']) for g,h in zip(grid,overlapping_grid(grid,fraction=.2)))
            X=grid[0]['X']; A=raw[order,a]; Y=data['Y'][order]
            rng=np.random.default_rng(8100000+10000*setting['seed']+rep)
            draws=[rng.integers(0,setting['n'],setting['n']) for _ in range(boots)]
            for method, designs in [('nonoverlap',grid),('overlap40',overlapping_grid(grid))]:
                row,Z,W,chosen=configuration_grid(designs,a)
                row.update(scenario=setting['name'],replicate=rep,method=method,n=setting['n'],p=system['p'],
                           true_r=system['r'],target=data['target'],boot_requested=boots,
                           **role_validity(graph,factors,y_node,a,Z,W))
                if row['status']=='reported':
                    estimate=proximal_bridge(Y,A,X[:,W],X[:,Z],row['r_bridge'],min_sv_ratio=1e-8)[0]
                    values=[]
                    for ix in draws:
                        value=proximal_bridge(Y[ix],A[ix],X[ix][:,W],X[ix][:,Z],row['r_bridge'],min_sv_ratio=1e-8)[0]
                        if np.isfinite(value): values.append(value)
                    se=float(np.std(values,ddof=1)) if len(values)>1 else np.nan
                    row.update(estimate=estimate,error=estimate-data['target'],se=se,
                               boot_success=len(values),covered=float(abs(estimate-data['target'])<=1.959963984540054*se),
                               rank_correct=row['r_bridge']==system['r'])
                records.append(row)
            if (rep+1)%10==0: print(setting['name'],rep+1,'elapsed',round(time.time()-start,1),flush=True)
        pd.DataFrame(records).to_csv(OUT/'window_prototype_simulation_replicates.csv',index=False)
    summaries=[]
    for (scenario,method), data in pd.DataFrame(records).groupby(['scenario','method']):
        fits=data[data.status.eq('reported')]
        summaries.append(dict(scenario=scenario,method=method,attempts=len(data),reported=len(fits),
                              report_rate=len(fits)/len(data),valid_roles=fits.set_valid.mean(),
                              rank_correct=fits.rank_correct.mean(),bias=fits.error.mean(),
                              rmse=np.sqrt(np.mean(fits.error**2)),
                              coverage=float(np.asarray(fits.covered,dtype=float).mean()) if len(fits) else np.nan,
                              bias_mcse=fits.error.std(ddof=1)/np.sqrt(len(fits)) if len(fits)>1 else np.nan))
    table=pd.DataFrame(summaries);table.to_csv(OUT/'window_prototype_simulation_summary.csv',index=False)
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reps',type=int,default=30)
    parser.add_argument('--boot',type=int,default=100);args=parser.parse_args()
    run(args.reps,args.boot)
